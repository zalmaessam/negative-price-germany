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


# ---------------------------------------------------------------------------
# "Morning information" features: only what is known BEFORE the auction (about 10:30 on D-1).
# The target day's own wind/solar forecasts are published later, so they are NOT used. Instead:
# calendar, prices up to D-1, and what the market ALREADY published for D-1 (persistence of weather).
# ---------------------------------------------------------------------------
MORNING_CAL = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"]
MORNING_HIST = ["price_lag_24", "price_lag_168", "neg_count_prev_day", "price_prev_day_mean"]
MORNING_PERSIST = ["res_lag24", "solar_lag24", "wind_lag24", "coupled_lag24",
                   "res_prevday_mean", "res_prevday_min", "solar_prevday_max", "coupled_prevday_mean"]
MORNING_LOAD = ["load_forecast"]   # the target day's load forecast: usually published earlier; verify with the monitor


def add_morning_features(df):
    """Persistence-of-yesterday features. Needs residual_load, solar_forecast, wind_total_forecast and
    (optionally) coupled_residual. Returns a DataFrame aligned to df.index.

    * *_lag24: the value 24 hours earlier, blanked when that moment lies inside the target day itself
      (the extra hour of a 25-hour day), exactly like the price lags.
    * *_prevday_*: summaries of the PREVIOUS local day's published forecasts.
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    cols = [c for c in ("residual_load", "solar_forecast", "wind_total_forecast", "coupled_residual") if c in df.columns]
    x = df[cols].reindex(full_idx)
    idx_local = full_idx.tz_convert(LOCAL_TZ)
    day_start_utc = idx_local.normalize().tz_convert("UTC")
    day_naive = idx_local.normalize().tz_localize(None)
    prev_day = day_naive - pd.Timedelta(days=1)

    def prev_agg(series, how):
        return series.groupby(day_naive).agg(how).reindex(prev_day).to_numpy()

    out = pd.DataFrame(index=full_idx)
    out["res_lag24"] = safe_lag(x["residual_load"], 24, day_start_utc)
    out["solar_lag24"] = safe_lag(x["solar_forecast"], 24, day_start_utc)
    out["wind_lag24"] = safe_lag(x["wind_total_forecast"], 24, day_start_utc)
    out["res_prevday_mean"] = prev_agg(x["residual_load"], "mean")
    out["res_prevday_min"] = prev_agg(x["residual_load"], "min")
    out["solar_prevday_max"] = prev_agg(x["solar_forecast"], "max")
    if "coupled_residual" in x:
        out["coupled_lag24"] = safe_lag(x["coupled_residual"], 24, day_start_utc)
        out["coupled_prevday_mean"] = prev_agg(x["coupled_residual"], "mean")
    else:  # no neighbour data: keep the columns so feature lists stay valid
        out["coupled_lag24"] = np.nan
        out["coupled_prevday_mean"] = np.nan
    return out.reindex(df.index)


def fill_neighbour_residual(data):
    """Forward-fill missing neighbour residuals from PAST values only and rebuild coupled_residual.

    Adds `nb_filled` = 1 where the value had to be filled, so results can be re-checked on the
    hours with complete neighbour data. Returns (data, number_of_filled_hours).
    Shared by the experiments, the final test run and the live prediction tool.
    """
    out = data.copy()
    out["nb_filled"] = out["nb_residual_sum"].isna().astype(int)
    out["nb_residual_sum"] = out["nb_residual_sum"].ffill()  # past values only: no leakage
    out["coupled_residual"] = out["residual_load"] + out["nb_residual_sum"]
    return out, int(out["nb_filled"].sum())


if __name__ == "__main__":
    data = make_dataset(force_rebuild=True)
    print("\nShape:", data.shape)
    print("\nNegative-hour rate by year (%):")
    print((data["target"].groupby(data.index.year).mean() * 100).round(2))
    print("\nFeature check (first rows):")
    print(data[ALL_FEATURES].head())