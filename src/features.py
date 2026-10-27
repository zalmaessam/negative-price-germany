"""
features.py  --  turns the raw hourly table into a leakage-safe modelling table.

THE RULE: every feature must be known at auction gate closure (about 12:00
Berlin time on the day before delivery). That means:
    - day-ahead FORECASTS for the delivery day are allowed,
    - prices are allowed only up to the end of the previous local day,
    - actual generation / actual weather for the delivery day is NOT allowed.
"""
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import get_energy_data  # noqa: E402

LOCAL_TZ = "Europe/Berlin"
FEATURES_FILE = Path("data") / "features.parquet"

# ---------------------------------------------------------------------------
# Feature lists. Logistic regression gets cyclical (sin/cos) encodings because
# it can't learn that hour 23 is next to hour 0; trees can use raw integers.
# ---------------------------------------------------------------------------
_SHARED = [
    "load_forecast", "solar_forecast", "wind_total_forecast",
    "residual_load", "res_share", "renewable_share",
    "is_weekend", "is_holiday",
    "price_lag_24", "price_lag_168",
    "neg_count_prev_day", "price_prev_day_mean",
]
LINEAR_FEATURES = _SHARED + ["hour_sin", "hour_cos", "month_sin", "month_cos"]
TREE_FEATURES = _SHARED + [
    "wind_onshore_forecast", "wind_offshore_forecast", "hour", "weekday", "month",
]
ALL_FEATURES = sorted(set(LINEAR_FEATURES) | set(TREE_FEATURES))

# Hard guard: these must never become model inputs
_FORBIDDEN = {"price", "neg_frac_target", "target"}
assert not (_FORBIDDEN & set(ALL_FEATURES)), "Target column leaked into feature list!"


# ---------------------------------------------------------------------------
# Helper: leakage-safe lag
# ---------------------------------------------------------------------------
def safe_lag(series: pd.Series, hours: int, day_start_utc: pd.DatetimeIndex) -> pd.Series:
    """Price from `hours` ago, but NaN if that moment is not yet known.

    A plain shift(24) is safe on normal days. On the 25-hour day in October the
    last hour of the day would look back to the FIRST hour of the same day,
    which is part of the very auction being predicted. This blanks those few
    rows. Knowledge cut-off for delivery day D = local midnight at the start of D.
    """
    lagged = series.shift(hours)
    source_time = series.index - pd.Timedelta(hours=hours)
    return lagged.mask(np.asarray(source_time >= day_start_utc))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target + features to the raw hourly table. Does NOT drop rows."""
    out = df.copy()
    idx_local = out.index.tz_convert(LOCAL_TZ)

    # ---- Section 1: TARGET -------------------------------------------------
    # 1 if the hourly mean price is negative, else 0. Missing price -> NaN.
    out["target"] = (out["price"] < 0).astype(float).where(out["price"].notna())

    # ---- Section 2: RESIDUAL LOAD FAMILY -----------------------------------
    # Residual load = demand the conventional plants must still cover after
    # wind and solar. Very low or negative residual load -> negative prices.
    out["wind_total_forecast"] = out["wind_onshore_forecast"] + out["wind_offshore_forecast"]
    out["residual_load"] = (
        out["load_forecast"] - out["solar_forecast"] - out["wind_total_forecast"]
    )
    load_safe = out["load_forecast"].where(out["load_forecast"] > 0)  # avoid /0
    out["res_share"] = out["residual_load"] / load_safe
    out["renewable_share"] = (out["solar_forecast"] + out["wind_total_forecast"]) / load_safe

    # ---- Section 3: CALENDAR FEATURES (local time) -------------------------
    # Demand follows the local clock, so we use Berlin time, not UTC.
    out["hour"] = idx_local.hour
    out["weekday"] = idx_local.day_of_week
    out["month"] = idx_local.month
    out["is_weekend"] = (idx_local.day_of_week >= 5).astype(int)

    # Cyclical encodings so that 23:00 and 00:00 (or Dec and Jan) are neighbours
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    # National German public holidays (regional holidays are not included)
    years = range(idx_local.year.min(), idx_local.year.max() + 1)
    de_holidays = holidays.Germany(years=years)
    local_dates = pd.Series(idx_local.date)
    holiday_by_date = {d: (d in de_holidays) for d in local_dates.unique()}
    out["is_holiday"] = local_dates.map(holiday_by_date).astype(int).to_numpy()

    # ---- Section 4: LAGGED PRICES (minimum lag 24 h, DST-safe) -------------
    # Start of each row's local delivery day, expressed in UTC.
    day_start_utc = idx_local.normalize().tz_convert("UTC")
    out["price_lag_24"] = safe_lag(out["price"], 24, day_start_utc)    # same hour yesterday
    out["price_lag_168"] = safe_lag(out["price"], 168, day_start_utc)  # same hour last week

    # ---- Section 5: PREVIOUS-DAY SUMMARIES ---------------------------------
    # Group by local calendar day, then attach the PREVIOUS day's value to each row.
    day_naive = idx_local.normalize().tz_localize(None)
    prev_day = day_naive - pd.Timedelta(days=1)

    daily_neg_count = out["target"].groupby(day_naive).sum(min_count=1)
    daily_mean_price = out["price"].groupby(day_naive).mean()
    out["neg_count_prev_day"] = daily_neg_count.reindex(prev_day).to_numpy()
    out["price_prev_day_mean"] = daily_mean_price.reindex(prev_day).to_numpy()

    return out


# ---------------------------------------------------------------------------
# Drop incomplete rows (only now, after every lag has been built)
# ---------------------------------------------------------------------------
def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with any missing feature or target, and log how many and why."""
    required = ALL_FEATURES + ["target"]
    bad = df[required].isna()
    print(f"Rows before drop: {len(df)}")
    print("Missing values per column (rows that will be dropped):")
    print(bad.sum()[bad.sum() > 0].to_string() or "  none")
    out = df.dropna(subset=required)
    print(f"Rows dropped: {len(df) - len(out)} | rows kept: {len(out)}")
    return out


def make_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """Load raw data -> build features -> drop incomplete rows -> cache."""
    if FEATURES_FILE.exists() and not force_rebuild:
        out = pd.read_parquet(FEATURES_FILE)
        out.index = pd.to_datetime(out.index).tz_convert("UTC")
        return out
    raw = get_energy_data()
    out = drop_incomplete(build_features(raw))
    out.to_parquet(FEATURES_FILE)
    return out


if __name__ == "__main__":
    data = make_dataset(force_rebuild=True)
    print("\nShape:", data.shape)
    print("\nNegative-hour rate by year (%):")
    print((data["target"].groupby(data.index.year).mean() * 100).round(2))
    print("\nFeature check (first rows):")
    print(data[ALL_FEATURES].head())