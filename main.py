from fastapi import FastAPI, Query, HTTPException
import polars as pl
import httpx
import re
from datetime import datetime, timezone
from cachetools import TTLCache

app = FastAPI(
    title="Quant API",
    description="High-performance, financial API for Quant Traders & AI Agents.",
    version="1.4.0"
)

# ---------------------------------------------------------
# In-Memory Cache (LRU + TTL)
# ---------------------------------------------------------
CACHE_TTL = 300        # Cache lifespan = 5 minutes
CACHE_MAX  = 500       # Maximum concurrent tickers
MEMORY_CACHE: TTLCache = TTLCache(maxsize=CACHE_MAX, ttl=CACHE_TTL)

# FIX #4: Whitelist valid data_range values
VALID_RANGES = {"1y", "2y", "5y", "10y", "max"}

# ---------------------------------------------------------
# Core: Fetch Data & Calculate Indicators (Async + Pure Polars)
# ---------------------------------------------------------
async def fetch_and_calculate(ticker: str, data_range: str = "10y") -> pl.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"interval": "1d", "range": data_range}
    headers = {"User-Agent": "Mozilla/5.0"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, headers=headers, timeout=10)

    if resp.status_code != 200:
        raise ValueError(f"Yahoo Finance API returned status {resp.status_code}. Please verify the ticker or try again later.")

    data = resp.json()
    if not data.get("chart", {}).get("result"):
        raise ValueError(f"Ticker '{ticker}' not found in Yahoo Finance database.")

    result = data["chart"]["result"][0]
    quotes = result["indicators"]["quote"][0]

    # Create DataFrame directly from JSON
    df = pl.DataFrame({
        "Date":   [datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                   for ts in result["timestamp"]],
        "Open":   quotes.get("open",   []),
        "High":   quotes.get("high",   []),
        "Low":    quotes.get("low",    []),
        "Close":  quotes.get("close",  []),
        "Volume": quotes.get("volume", []),
    })

    # Handle Yahoo's null gaps (forward-fill then drop leading nulls)
    df = df.fill_null(strategy="forward").drop_nulls()

    # --- Block 1: Base Differences ---
    df = df.with_columns([
        pl.col("Close").diff().alias("price_diff"),
        pl.col("Close").shift(1).alias("prev_close"),
    ])

    # --- Block 2: SMA, EMA, BB Std, MACD base, True Range ---
    df = df.with_columns([
        pl.col("Close").rolling_mean(window_size=20).alias("SMA_20"),
        pl.col("Close").ewm_mean(span=50,  adjust=False).alias("EMA_50"),
        pl.col("Close").rolling_std(window_size=20).alias("BB_Std"),
        pl.col("Close").ewm_mean(span=12,  adjust=False).alias("EMA_12"),
        pl.col("Close").ewm_mean(span=26,  adjust=False).alias("EMA_26"),
        pl.max_horizontal([
            (pl.col("High") - pl.col("Low")),
            (pl.col("High") - pl.col("prev_close")).abs(),
            (pl.col("Low")  - pl.col("prev_close")).abs(),
        ]).alias("True_Range"),
    ])

    # --- Block 3: BB, MACD Line, Gain/Loss, OBV Step ---
    df = df.with_columns([
        (pl.col("SMA_20") + pl.col("BB_Std") * 2).alias("BB_Upper"),
        (pl.col("SMA_20") - pl.col("BB_Std") * 2).alias("BB_Lower"),
        (pl.col("EMA_12") - pl.col("EMA_26")).alias("MACD_Line"),
        pl.when(pl.col("price_diff") > 0)
          .then(pl.col("price_diff"))
          .otherwise(0).alias("gain"),
        pl.when(pl.col("price_diff") < 0)
          .then(pl.col("price_diff").abs())
          .otherwise(0).alias("loss"),
        pl.when(pl.col("price_diff") > 0).then(pl.col("Volume"))
          .when(pl.col("price_diff") < 0).then(-pl.col("Volume"))
          .otherwise(0).alias("OBV_Step"),
    ])

    # --- Block 4: Signal, Avg Gain/Loss (Wilder's EWM), ATR, OBV ---
    # FIX #2: Use Wilder's Smoothing (alpha=1/14) instead of rolling_mean for RSI
    df = df.with_columns([
        pl.col("MACD_Line").ewm_mean(span=9, adjust=False).alias("MACD_Signal"),
        pl.col("gain").ewm_mean(alpha=1/14, adjust=False).alias("avg_gain"),   # ✅ Fixed
        pl.col("loss").ewm_mean(alpha=1/14, adjust=False).alias("avg_loss"),   # ✅ Fixed
        pl.col("True_Range").ewm_mean(alpha=1/14, adjust=False).alias("ATR_14"),
        pl.col("OBV_Step").cum_sum().alias("OBV"),
    ])

    # --- Block 5: MACD Hist + RSI (div-by-zero safe) ---
    safe_avg_loss = pl.when(pl.col("avg_loss") == 0).then(1e-10).otherwise(pl.col("avg_loss"))

    df = df.with_columns([
        (pl.col("MACD_Line") - pl.col("MACD_Signal")).alias("MACD_Hist"),
        (100 - (100 / (1 + (pl.col("avg_gain") / safe_avg_loss)))).alias("RSI_14"),
    ])

    # FIX #1: Drop nulls AFTER all indicators are calculated
    return df.select([
        "Date", "Close", "Volume",
        "SMA_20", "EMA_50", "RSI_14",
        "MACD_Line", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "ATR_14", "OBV",
    ]).drop_nulls()  # ✅ Fixed


# ---------------------------------------------------------
# Endpoint: GET /api/v1/indicators/{ticker}
# ---------------------------------------------------------
@app.get("/api/v1/indicators/{ticker}")
async def get_quant_data(
    ticker: str,
    target_days: int = Query(10, ge=1, le=5000, description="Number of trading days to retrieve (1-5000)"),
    data_range: str = Query("10y", description="Raw data range to fetch for indicator warm-up (e.g., 1y, 2y, 5y, 10y, max)")
):
    ticker = ticker.upper()

    # --- Validate ticker format ---
    if not re.match(r"^[A-Z0-9.\-]{1,10}$", ticker):
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker format. Only uppercase letters, numbers, dots, and hyphens are allowed (max 10 characters)."
        )

    # FIX #4: Validate data_range against whitelist
    if data_range not in VALID_RANGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data_range '{data_range}'. Must be one of: {sorted(VALID_RANGES)}"
        )

    # FIX #3: Use '::' separator to avoid cache key collision
    cache_key = f"{ticker}::{data_range}"
    served_from_cache = cache_key in MEMORY_CACHE

    if served_from_cache:
        print(f"⚡ [CACHE HIT]  {cache_key}")
        df_clean = MEMORY_CACHE[cache_key]
    else:
        print(f"🌐 [CACHE MISS] {cache_key} — Fetching from Yahoo...")
        try:
            df_clean = await fetch_and_calculate(ticker, data_range)
            MEMORY_CACHE[cache_key] = df_clean
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Yahoo Finance API request timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

    # --- Slice data to target_days ---
    final_df = df_clean.tail(target_days)

    return {
        "metadata": {
            "ticker":            ticker,
            "target_days":       target_days,
            "data_range_used":   data_range,
            "rows_returned":     len(final_df),
            "served_from_cache": served_from_cache,
            "timestamp":         datetime.now(tz=timezone.utc).isoformat(),
        },
        "data": final_df.to_dict(as_series=False),
    }
