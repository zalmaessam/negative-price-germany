import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CACHE_FILE = DATA_DIR / "de_lu_energy_data.parquet"

COUNTRY_CODE = "DE_LU"
START = pd.Timestamp("2019-01-01 00:00:00", tz="UTC")
END = pd.Timestamp("2026-10-01 00:00:00", tz="UTC")  # exclusive in ENTSO-E

# Columns that are NOT features (price is the target source, neg_frac_target is
# a secondary target). Never pass these to a model as inputs.
TARGET_COLS = ("price", "neg_frac_target")

# Expected last timestamp in a correctly built cache (30 Sept 2026, 23:00 UTC)
EXPECTED_LAST = pd.Timestamp("2026-09-30 23:00:00", tz="UTC")


def _cache_is_valid(df: pd.DataFrame) -> bool:
    """A cache is valid only if it has the new column and the extended range."""
    if "neg_frac_target" not in df.columns:
        print("Cache check failed: 'neg_frac_target' missing (old cache).")
        return False
    if df.index.max() < EXPECTED_LAST:
        print(f"Cache check failed: ends {df.index.max()}, expected {EXPECTED_LAST}.")
        return False
    return True


def get_feature_columns(df: pd.DataFrame) -> list:
    """Single place that defines which columns are allowed as model inputs."""
    return [c for c in df.columns if c not in TARGET_COLS]


def get_energy_data(force_download: bool = False) -> pd.DataFrame:
    """
    Load DE_LU day-ahead price and day-ahead forecast data (hourly, UTC).

    Behaviour
    ---------
    - Uses the local parquet cache if it exists AND passes validation
      (has 'neg_frac_target', ends on 2026-09-30 23:00 UTC). Otherwise refetches.
    - Resamples everything to hourly means in UTC.
    - Reindexes to a complete hourly grid so that row-based shifts
      (shift(24), shift(168)) equal time-based shifts.
    - Does NOT drop rows with missing prices or features. Drop them in
      features.py only AFTER lag features are built.

    Methods notes
    -------------
    - Market resolution: the European day-ahead market moved to 15-minute
      products on 1 Oct 2025. Primary target: hourly mean price < 0.
    - 'neg_frac_target' = share of non-missing quarter-hours with price < 0
      within each hour. Before Oct 2025 it can only be 0 or 1 (one price per
      hour); afterwards it can be 0, 0.25, 0.5, 0.75 or 1. It is a TARGET,
      never a feature.
    - Leakage: only day-ahead forecasts are used. The load forecast is
      normally published before gate closure (~12:00 CET on D-1). The
      wind/solar day-ahead forecast publication time for DE-LU must be
      verified on the ENTSO-E Transparency Platform. If it is later than
      gate closure, state the claim as "forecasts available before the
      delivery day" and list it as a limitation.
    """
    # 1. CACHE CHECK
    if CACHE_FILE.exists() and not force_download:
        print(f"Loading data from local cache: {CACHE_FILE}")
        df = pd.read_parquet(CACHE_FILE)
        df.index = pd.to_datetime(df.index).tz_convert("UTC")
        df.index.name = "timestamp_utc"
        if _cache_is_valid(df):
            return df
        print("Rebuilding cache from the ENTSO-E API...")

    print("Fetching raw data from ENTSO-E API...")

    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOES_API_KEY")
    if not api_key:
        raise ValueError("API key not found! Check your .env file for ENTSOE_API_KEY.")

    client = EntsoePandasClient(api_key=api_key)

    # 2. QUERY RAW DATA
    prices = client.query_day_ahead_prices(COUNTRY_CODE, start=START, end=END)
    load_forecast = client.query_load_forecast(COUNTRY_CODE, start=START, end=END)
    wind_solar_forecast = client.query_wind_and_solar_forecast(
        COUNTRY_CODE, start=START, end=END
    )

    print("Standardizing resolutions to hourly mean format...")

    # Strict UTC normalization
    prices = prices.tz_convert("UTC")
    load_forecast = load_forecast.tz_convert("UTC")
    wind_solar_forecast = wind_solar_forecast.tz_convert("UTC")

    # 3. RESOLUTION HANDLING
    # Secondary target: cast to float BEFORE masking so NaN is representable
    # (a bool Series cannot hold NaN and would fall back to object dtype).
    neg_frac_target = (
        (prices < 0).astype(float).where(prices.notna()).resample("h").mean()
    )

    prices_h = prices.resample("h").mean()
    load_h = load_forecast.resample("h").mean()
    wind_solar_h = wind_solar_forecast.resample("h").mean()

    # 4. INSPECT MISSING VALUES (raw, before gridding)
    print("\n--- Initial NaN Inspection ---")
    print("Prices NaNs:", int(prices_h.isna().sum()))
    print("Load Forecast NaNs:\n", load_h.isna().sum())
    print("Wind/Solar Forecast NaNs:\n", wind_solar_h.isna().sum())

    # 5. COMBINE
    df = pd.DataFrame({"price": prices_h, "neg_frac_target": neg_frac_target})

    df["load_forecast"] = load_h if isinstance(load_h, pd.Series) else load_h.iloc[:, 0]

    if isinstance(wind_solar_h, pd.DataFrame):
        for col in wind_solar_h.columns:
            clean_col_name = str(col).lower().replace(" ", "_")
            df[f"{clean_col_name}_forecast"] = wind_solar_h[col]
    else:
        df["wind_solar_forecast"] = wind_solar_h

    # 6. FULL HOURLY GRID (row shifts == time shifts)
    full_idx = pd.date_range(START, EXPECTED_LAST, freq="h", tz="UTC")
    df = df.reindex(full_idx)
    df.index.name = "timestamp_utc"

    # 7. FEATURE GAP FILLING: forward-fill only, short gaps only (no future info)
    feature_cols = get_feature_columns(df)
    df[feature_cols] = df[feature_cols].ffill(limit=3)

    # Report, but do not drop: lags must be built on the full grid first
    print("\nMissing price hours:", int(df["price"].isna().sum()))
    print(
        "Rows with unresolved feature NaNs:",
        int(df[feature_cols].isna().any(axis=1).sum()),
    )
    assert df.index.is_monotonic_increasing and df.index.is_unique

    # 8. SAVE CACHE (parquet keeps UTC metadata)
    print(f"Saving processed data to local cache: {CACHE_FILE}")
    df.to_parquet(CACHE_FILE)

    return df


if __name__ == "__main__":
    # The cache is validated automatically, so a stale file is rebuilt on its own.
    # Use force_download=True only if you want to refetch deliberately.
    data = get_energy_data(force_download=False)

    print("\nData successfully loaded!")
    print(data.info())
    print("\nIndex range:", data.index.min(), "->", data.index.max(), "| rows:", len(data))
    print("\nFeature columns:", get_feature_columns(data))
    print("\nMissing values per year:")
    print(data.isna().groupby(data.index.year).sum())
    print("\nFirst 5 rows:")
    print(data.head())