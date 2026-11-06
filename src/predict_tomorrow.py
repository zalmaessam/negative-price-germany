"""
predict_tomorrow.py  --  day-before negative-price early-warning forecast for Germany-Luxembourg.

    python src/predict_tomorrow.py                      # LIVE: tomorrow, forecasts fetched from ENTSO-E
    python src/predict_tomorrow.py --date 2026-11-05    # LIVE for a specific target day
    python src/predict_tomorrow.py --replay 2026-09-12  # REPLAY a past day from the local cache (no API)
    python src/predict_tomorrow.py --verify 2025-07-15  # replay + compare with the frozen test-run predictions
    python src/predict_tomorrow.py --score              # fill in realised prices and print running accuracy
    python src/predict_tomorrow.py --check-availability # which inputs for tomorrow are published right now?
    python src/predict_tomorrow.py --schedule           # today's check times in Berlin and Cairo time
    python src/predict_tomorrow.py --replay 2026-06-20 --scenario "solar=1.4,wind=0.7"   # what-if (not saved)

What it does (the frozen configuration validated in final.py):
  1. Gets day-ahead FORECASTS of load, wind and solar for Germany and 9 neighbours, plus the
     previous days' prices. Prices of the target day itself are always masked out.
  2. Builds exactly the training features (features.py / neighbors.py code, no copies).
  3. Trains a logistic regression on the 365 days before the start of the target day's month
     (monthly refit, as validated), so there is no stored model that can go stale.
  4. Outputs hourly probabilities of a negative price, a day summary, per-hour drivers and a
     data-status report, as JSON/CSV for the dashboard.

Honesty rules built in:
  * A forecast generated AFTER 12:00 Berlin time on the day before delivery is marked as not
    prospective (the real prices may already be public).
  * Missing neighbour inputs are forward-filled from the past; the status says how many hours.
  * The prediction log is append-only: a logged hour is never overwritten.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import get_energy_data  # noqa: E402
from evaluation import evaluate  # noqa: E402
from features import ALL_FEATURES, LINEAR_FEATURES, build_features, fill_neighbour_residual  # noqa: E402
import neighbors  # noqa: E402
from neighbors import COUNTRIES, _retry, build_neighbor_features, fetch_country  # noqa: E402

LOCAL_TZ = "Europe/Berlin"
TRAIN_WINDOW_DAYS = 365
LOOKBACK_DAYS = 35               # extra history so lags and gap-filling match training exactly
DEFAULT_THRESHOLD = 0.5          # equal cost for a miss and a false alarm (the validated rule)
FROZEN_EXCLUDED = {"PL_solar"}   # dropped in training: only 84% coverage
DE_FORECAST_COLS = ["load_forecast", "solar_forecast", "wind_offshore_forecast", "wind_onshore_forecast"]
PRICE_LAG_COLS = ["price_lag_24", "price_lag_168", "neg_count_prev_day", "price_prev_day_mean"]

OUT_DIR = Path("outputs")
FORECAST_DIR = OUT_DIR / "forecasts"
LOG_FILE = OUT_DIR / "prediction_log.csv"
LIVE_CACHE = Path("data") / "live"


# ---------------------------------------------------------------------------
# Section 1: dates and windows
# ---------------------------------------------------------------------------
def day_bounds(target_date):
    """Local calendar day -> (start_utc, end_utc). Handles 23- and 25-hour DST days."""
    d = pd.Timestamp(target_date)
    start = pd.Timestamp(d.date().isoformat(), tz=LOCAL_TZ).tz_convert("UTC")
    end = pd.Timestamp((d + pd.Timedelta(days=1)).date().isoformat(), tz=LOCAL_TZ).tz_convert("UTC")
    return start, end


def month_start_utc(target_date):
    """Model 'as of' time: local midnight on the 1st of the target day's month (monthly refit)."""
    d = pd.Timestamp(target_date)
    return pd.Timestamp(f"{d.year}-{d.month:02d}-01", tz=LOCAL_TZ).tz_convert("UTC")


def window_grid(target_date):
    as_of = month_start_utc(target_date)
    _, t_end = day_bounds(target_date)
    w_start = as_of - pd.Timedelta(days=TRAIN_WINDOW_DAYS + LOOKBACK_DAYS)
    return w_start, pd.date_range(w_start, t_end - pd.Timedelta(hours=1), freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# Section 2: data sources (live API or local cache)
# ---------------------------------------------------------------------------
def _client():
    from entsoe import EntsoePandasClient

    load_dotenv()
    key = os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOES_API_KEY")
    if not key:
        raise ValueError("API key not found in .env (ENTSOE_API_KEY).")
    return EntsoePandasClient(api_key=key)


def _hourly(obj):
    return obj.tz_convert("UTC").resample("h").mean()


def fetch_de(client, start, end, grid):
    """German price + day-ahead forecasts, hourly UTC, same column names as data_loader."""
    prices = _hourly(_retry(lambda: client.query_day_ahead_prices("DE_LU", start=start, end=end)))
    load = _hourly(_retry(lambda: client.query_load_forecast("DE_LU", start=start, end=end)))
    ws = _hourly(_retry(lambda: client.query_wind_and_solar_forecast("DE_LU", start=start, end=end)))
    df = pd.DataFrame({"price": prices})
    df["load_forecast"] = load if isinstance(load, pd.Series) else load.iloc[:, 0]
    for col in ws.columns:
        df[f"{str(col).lower().replace(' ', '_')}_forecast"] = ws[col]
    missing = [c for c in DE_FORECAST_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"German forecast columns missing from the API response: {missing}")
    return df.reindex(grid)


def fetch_live_raw(target_date, refresh=False):
    """Download the window (history + target day). Cached per target date in data/live/."""
    LIVE_CACHE.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(target_date).date().isoformat()
    f_de, f_nb, f_av = (LIVE_CACHE / f"de_{tag}.parquet", LIVE_CACHE / f"nb_{tag}.parquet",
                        LIVE_CACHE / f"avail_{tag}.json")
    if f_de.exists() and f_nb.exists() and f_av.exists() and not refresh:
        de, nb = pd.read_parquet(f_de), pd.read_parquet(f_nb)
        de.index, nb.index = pd.to_datetime(de.index, utc=True), pd.to_datetime(nb.index, utc=True)
        return de, nb, json.load(open(f_av))

    t_start, t_end = day_bounds(target_date)
    w_start, grid = window_grid(target_date)
    client = _client()
    print(f"Fetching Germany {w_start.date()} .. {t_end.date()} ...")
    de = fetch_de(client, w_start, t_end, grid)

    frames = []
    for cc in COUNTRIES:
        print(f"Fetching {cc} ...")
        df = fetch_country(client, cc, start=w_start, end=t_end)
        df.columns = [f"{cc}_{c}" for c in df.columns]
        frames.append(df)
    nb = pd.concat(frames, axis=1).reindex(grid) if frames else pd.DataFrame(index=grid)

    # What was actually published for the target day, BEFORE any gap-filling
    day = (grid >= t_start) & (grid < t_end)
    avail = {
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "de_missing_hours": {c: int(de.loc[day, c].isna().sum()) for c in DE_FORECAST_COLS},
        "nb_missing_hours": {c: int(nb.loc[day, c].isna().sum()) for c in nb.columns},
        "last_valid_utc": {c: (None if s.last_valid_index() is None else s.last_valid_index().isoformat())
                           for c, s in {**{c: de[c] for c in DE_FORECAST_COLS}, **{c: nb[c] for c in nb.columns}}.items()},
    }
    # Same short-gap fill as the training loaders (forward fill, at most 3 hours)
    de[DE_FORECAST_COLS] = de[DE_FORECAST_COLS].ffill(limit=3)
    nb = nb.ffill(limit=3)
    de.to_parquet(f_de)
    nb.to_parquet(f_nb)
    json.dump(avail, open(f_av, "w"), indent=2)
    return de, nb, avail


def load_replay_raw(target_date):
    """Slice the local training caches (already gap-filled by the loaders)."""
    t_start, t_end = day_bounds(target_date)
    w_start, grid = window_grid(target_date)
    de = get_energy_data()
    de.index = pd.to_datetime(de.index, utc=True)
    if de.index.min() > w_start or de.index.max() < t_end - pd.Timedelta(hours=1):
        raise SystemExit(f"{target_date} is outside the cached data "
                         f"({de.index.min().date()} .. {de.index.max().date()}). Use live mode for newer days.")
    nb = pd.read_parquet(neighbors.NEIGHBORS_FILE)
    nb.index = pd.to_datetime(nb.index, utc=True)
    return de.reindex(grid), nb.reindex(grid), {}


# ---------------------------------------------------------------------------
# What-if scenarios: the user supplies their own expectation about tomorrow's weather
# ---------------------------------------------------------------------------
SCENARIO_DEFAULT = {"solar": 1.0, "wind": 1.0, "load": 1.0, "nb_solar": 1.0, "nb_wind": 1.0, "nb_load": 1.0}


def parse_scenario(text):
    """'solar=1.3,wind=0.8' -> dict with defaults for everything not given."""
    out = dict(SCENARIO_DEFAULT)
    for part in text.split(","):
        if not part.strip():
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in SCENARIO_DEFAULT:
            raise SystemExit(f"Unknown scenario key '{key}'. Allowed: {', '.join(SCENARIO_DEFAULT)}")
        out[key] = float(val)
    return out


def apply_scenario(raw_de, raw_nb, target_date, scenario):
    """Replace the target day's forecast inputs by the PREVIOUS day's published values times the user's factors.

    Hours are matched by local clock hour, so clock-change days work. Nothing published for the target day
    is used: the scenario is entirely 'yesterday, scaled by your expectation'.
    """
    sc = {**SCENARIO_DEFAULT, **scenario}
    t_start, t_end = day_bounds(target_date)
    prev = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).date().isoformat()
    p_start, p_end = day_bounds(prev)
    de, nb = raw_de.copy(), raw_nb.copy()
    tgt = de.index[(de.index >= t_start) & (de.index < t_end)]
    prv = de.index[(de.index >= p_start) & (de.index < p_end)]
    first_by_hour = {}
    for ts in prv:
        first_by_hour.setdefault(ts.tz_convert(LOCAL_TZ).strftime("%H"), ts)
    src, last = [], prv[0]
    for ts in tgt:
        last = first_by_hour.get(ts.tz_convert(LOCAL_TZ).strftime("%H"), last)
        src.append(last)
    de_mult = {"load_forecast": sc["load"], "solar_forecast": sc["solar"],
               "wind_onshore_forecast": sc["wind"], "wind_offshore_forecast": sc["wind"]}
    for col, m in de_mult.items():
        de.loc[tgt, col] = de.loc[src, col].to_numpy() * m
    for col in nb.columns:
        m = sc["nb_load"] if col.endswith("_load") else sc["nb_solar"] if col.endswith("_solar") else sc["nb_wind"]
        nb.loc[tgt, col] = nb.loc[src, col].to_numpy() * m
    return de, nb


# ---------------------------------------------------------------------------
# Section 3: features, model, prediction
# ---------------------------------------------------------------------------
def build_frame(raw_de, raw_nb, t_start):
    """Training-identical feature table for the whole window. Target-day prices are masked."""
    de = raw_de.copy()
    de.loc[de.index >= t_start, "price"] = np.nan          # never use prices of the target day
    base = build_features(de)

    used = {c for c in raw_nb.columns if c not in FROZEN_EXCLUDED and raw_nb[c].notna().any()}
    nb, nb_cols = build_neighbor_features(raw_nb, used=used)
    df = base.join(nb.reindex(base.index))
    df["coupled_residual"] = df["residual_load"] + df["nb_residual_sum"]
    df["coupled_res_share"] = df["coupled_residual"] / (df["load_forecast"] + df["nb_load_sum"])
    df, _ = fill_neighbour_residual(df)                      # past-only fill, adds nb_filled
    feats = LINEAR_FEATURES + nb_cols + ["coupled_residual", "coupled_res_share"]
    return df, feats, used


_EXPECTED = None


def expected_series():
    """Neighbour series the validated model was built on (training cache columns minus exclusions)."""
    global _EXPECTED
    if _EXPECTED is None:
        try:
            cols = pd.read_parquet(neighbors.NEIGHBORS_FILE).columns
            _EXPECTED = set(cols) - FROZEN_EXCLUDED
        except Exception:  # noqa: BLE001  (no cache: skip the check)
            _EXPECTED = set()
    return _EXPECTED


def frozen_C():
    path = Path("results") / "run_summary_val.json"
    return json.load(open(path))["LR"]["params"]["C"] if path.exists() else 0.1


def make_model(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def train_model(df, as_of, feats):
    """Fit on the 365 days before `as_of` (complete rows only), exactly like the validated rolling fit."""
    pool = df[(df.index < as_of) & (df.index >= as_of - pd.Timedelta(days=TRAIN_WINDOW_DAYS))]
    pool = pool.dropna(subset=ALL_FEATURES + ["target"])
    n_pos = int(pool["target"].sum())
    if len(pool) < 2000 or n_pos < 20:
        raise RuntimeError(f"Not enough training data ({len(pool)} rows, {n_pos} negative hours).")
    C = frozen_C()
    model = make_model(C).fit(pool[feats], pool["target"].astype(int))
    info = {"type": "logistic regression", "trained_as_of": as_of.isoformat(), "window_days": TRAIN_WINDOW_DAYS,
            "C": C, "n_train": int(len(pool)), "n_train_negative": n_pos,
            "train_last_hour_utc": pool.index.max().isoformat(), "features": feats}
    return model, info


def top_drivers(model, X, k=3):
    """Per-hour top-k contributions (standardised value x coefficient) to the negative-price score."""
    contrib = model[:-1].transform(X) * model[-1].coef_[0]
    out = []
    for row in contrib:
        idx = np.argsort(-np.abs(row))[:k]
        out.append([{"feature": str(X.columns[i]), "contribution": round(float(row[i]), 3)} for i in idx])
    return out


def flag_windows(local_idx, flags, probs):
    """Contiguous runs of flagged hours -> list of {start, end, max_probability}."""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if start is not None and (not f or i == len(flags) - 1):
            stop = i if f else i - 1
            runs.append({"start_local": local_idx[start].strftime("%H:%M"),
                         "end_local": (local_idx[stop] + pd.Timedelta(hours=1)).strftime("%H:%M"),
                         "max_probability": round(float(np.max(probs[start:stop + 1])), 3)})
            start = None
    return runs


def compute_status(day, used, avail):
    """Summarise input quality for the target day."""
    filled = int(day["nb_filled"].sum())
    nb_gaps = {k: v for k, v in avail.get("nb_missing_hours", {}).items() if v > 3 and k in used}
    de_gaps = {k: v for k, v in avail.get("de_missing_hours", {}).items() if v > 0}
    missing_series = sorted(expected_series() - used)
    if missing_series or nb_gaps or filled > 3:
        level = "degraded"
    elif filled > 0:
        level = "minor_gaps"
    else:
        level = "ok"
    notes = []
    if missing_series:
        notes.append("Validated neighbour series unavailable in this run: " + ", ".join(missing_series) +
                     ". The model was trained without them, so it differs from the validated configuration.")
    if filled:
        notes.append(f"{filled} hour(s): neighbour inputs incomplete, last known values were carried forward.")
    if nb_gaps:
        notes.append("Neighbour series with more than 3 unpublished hours: " + ", ".join(sorted(nb_gaps)))
    if de_gaps:
        notes.append("German inputs had short gaps that were forward-filled: " + ", ".join(sorted(de_gaps)))
    return {"level": level, "neighbour_hours_filled": filled, "neighbour_series_used": sorted(used),
            "neighbour_series_excluded": sorted(FROZEN_EXCLUDED), "neighbour_series_missing": missing_series,
            "notes": notes, "published_before_fill": avail}


# ---------------------------------------------------------------------------
# Section 4: one forecast
# ---------------------------------------------------------------------------
def forecast_day(target_date, source="api", refresh=False, threshold=DEFAULT_THRESHOLD, scenario=None):
    target_date = pd.Timestamp(target_date).date().isoformat()
    t_start, t_end = day_bounds(target_date)
    as_of = month_start_utc(target_date)
    raw_de, raw_nb, avail = (fetch_live_raw(target_date, refresh) if source == "api"
                             else load_replay_raw(target_date))
    if scenario:
        raw_de, raw_nb = apply_scenario(raw_de, raw_nb, target_date, scenario)
        avail = {}
    df, feats, used = build_frame(raw_de, raw_nb, t_start)
    model, info = train_model(df, as_of, feats)

    day = df[(df.index >= t_start) & (df.index < t_end)]
    # 1) forecast-based inputs must be complete for every hour
    bad = {c: int(day[c].isna().sum()) for c in LINEAR_FEATURES
           if c not in PRICE_LAG_COLS and day[c].isna().any()}
    if bad:
        now_berlin = pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        raise RuntimeError(
            "Cannot forecast: required forecasts are not available for the target day "
            f"(hours missing: {json.dumps(bad)}). Time now in Berlin: {now_berlin}. "
            "Wind/solar day-ahead forecasts may be published late (the EU deadline is 18:00 Brussels time on the "
            "day before). Run `python src/predict_tomorrow.py --check-availability` to see what is published, "
            "and see data/live/avail_<date>.json for the last published hour of every series.")
    # 2) price-lag inputs: only the extra hour of a 25-hour day may lack one (its 24-hour lag would
    #    fall inside the same auction). Anything more means yesterday's prices are missing.
    lag_missing = day[PRICE_LAG_COLS].isna().any(axis=1).to_numpy()
    if lag_missing.sum() > 1:
        raise RuntimeError("Cannot forecast: yesterday's prices are missing (price lags incomplete for "
                           f"{int(lag_missing.sum())} hours). Check the ENTSO-E price download.")

    X = day[feats]
    ok = ~lag_missing
    prob = np.full(len(day), np.nan)
    prob[ok] = model.predict_proba(X[ok])[:, 1]
    local = day.index.tz_convert(LOCAL_TZ)
    flags = np.where(ok, np.nan_to_num(prob) >= threshold, False)
    drivers = [[] for _ in range(len(day))]
    for i, d in zip(np.flatnonzero(ok), top_drivers(model, X[ok])):
        drivers[i] = d

    now = pd.Timestamp.now(tz="UTC")
    gate = pd.Timestamp((pd.Timestamp(target_date) - pd.Timedelta(days=1)).date().isoformat() + " 12:00", tz=LOCAL_TZ)
    before_gate = bool(now < gate) if source == "api" and not scenario else None
    status = compute_status(day, used, avail)
    if scenario:
        changed = ", ".join(f"{k} x{v:g}" for k, v in {**SCENARIO_DEFAULT, **scenario}.items() if v != 1.0) or "no changes"
        status["notes"].insert(0, "WHAT-IF SCENARIO, NOT A FORECAST: the target day's inputs are yesterday's published "
                                  f"values scaled by your factors ({changed}).")
    elif source == "api" and not before_gate:
        status["notes"].append("Generated after 12:00 Berlin time on the day before: the real prices may already "
                               "be public, so this forecast is NOT prospective.")

    if lag_missing.any():
        status["notes"].append("1 hour has no forecast: the extra hour of a 25-hour (clock-change) day has no "
                               "valid 24-hour price lag. The model was only validated with complete inputs.")
    hours = []
    for i, ts in enumerate(day.index):
        hours.append({
            "hour_utc": ts.isoformat(), "hour_local": local[i].strftime("%H:%M"),
            "probability": None if np.isnan(prob[i]) else round(float(prob[i]), 4),
            "flag": None if np.isnan(prob[i]) else bool(flags[i]),
            "load_mw": round(float(day["load_forecast"].iloc[i])), "solar_mw": round(float(day["solar_forecast"].iloc[i])),
            "wind_mw": round(float(day["wind_total_forecast"].iloc[i])),
            "residual_load_mw": round(float(day["residual_load"].iloc[i])),
            "coupled_residual_mw": round(float(day["coupled_residual"].iloc[i])),
            "drivers": drivers[i],
        })
    record = {
        "target_date": target_date, "source": f"scenario_{source}" if scenario else source,
        "scenario": ({**SCENARIO_DEFAULT, **scenario, "template_day":
                      (pd.Timestamp(target_date) - pd.Timedelta(days=1)).date().isoformat()} if scenario else None),
        "generated_at_utc": now.isoformat(),
        "before_gate_closure": before_gate, "default_threshold": threshold, "model": info, "status": status,
        "summary": {"expected_negative_hours": round(float(np.nansum(prob)), 2),
                    "max_probability": round(float(np.nanmax(prob)), 3), "hours_flagged": int(flags.sum()),
                    "hours_without_forecast": int(lag_missing.sum()),
                    "flagged_windows": flag_windows(local, flags, np.nan_to_num(prob))},
        "hours": hours,
    }
    return record


def save_record(record):
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    tag = record["target_date"]
    prefix = "forecast" if record["source"] == "api" else "replay"   # replays never touch the live files
    path = FORECAST_DIR / f"{prefix}_{tag}.json"
    json.dump(record, open(path, "w"), indent=2)
    if record["source"] == "api":
        json.dump(record, open(FORECAST_DIR / "latest.json", "w"), indent=2)
    pd.DataFrame([{k: v for k, v in h.items() if k != "drivers"} for h in record["hours"]]).to_csv(
        FORECAST_DIR / f"{prefix}_{tag}.csv", index=False)
    return path


# ---------------------------------------------------------------------------
# Section 5: append-only prediction log and scoring
# ---------------------------------------------------------------------------
def append_log(record):
    """Log hourly probabilities. Existing (date, hour) rows are never overwritten."""
    OUT_DIR.mkdir(exist_ok=True)
    new = pd.DataFrame({
        "target_date": record["target_date"],
        "hour_utc": [h["hour_utc"] for h in record["hours"]],
        "hour_local": [h["hour_local"] for h in record["hours"]],
        "probability": [np.nan if h["probability"] is None else h["probability"] for h in record["hours"]],
        "generated_at_utc": record["generated_at_utc"],
        "before_gate_closure": record["before_gate_closure"],
        "model_as_of": record["model"]["trained_as_of"],
        "status_level": record["status"]["level"],
        "actual_price": np.nan, "actual_negative": np.nan,
    })
    if LOG_FILE.exists():
        old = pd.read_csv(LOG_FILE)
        seen = set(zip(old["target_date"].astype(str), old["hour_utc"]))
        new = new[[(d, h) not in seen for d, h in zip(new["target_date"].astype(str), new["hour_utc"])]]
        if new.empty:
            print("This day is already in the log; existing rows are kept (append-only).")
            return
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(LOG_FILE, index=False)
    print(f"Logged to {LOG_FILE}")


def score_log(threshold=DEFAULT_THRESHOLD):
    """Fill in realised prices for finished days and print running accuracy."""
    if not LOG_FILE.exists():
        raise SystemExit("No prediction log yet. Run a live forecast first.")
    log = pd.read_csv(LOG_FILE)
    log["hour_ts"] = pd.to_datetime(log["hour_utc"], utc=True)
    today = pd.Timestamp.now(tz=LOCAL_TZ).date().isoformat()
    pending = log[log["actual_price"].isna() & (log["target_date"].astype(str) < today)]
    if len(pending):
        client = _client()
        prices = _hourly(client.query_day_ahead_prices(
            "DE_LU", start=pending["hour_ts"].min(), end=pending["hour_ts"].max() + pd.Timedelta(hours=1)))
        got = log["hour_ts"].map(prices)
        fill = log["actual_price"].isna() & got.notna()
        log.loc[fill, "actual_price"] = got[fill]
        log.loc[fill, "actual_negative"] = (got[fill] < 0).astype(float)
        print(f"Filled outcomes for {int(fill.sum())} hours.")
    log.drop(columns="hour_ts").to_csv(LOG_FILE, index=False)

    done = log[log["actual_price"].notna() & log["probability"].notna()]
    if done.empty:
        print("No finished days with published prices yet.")
        return
    for label, sub in [("all logged hours", done),
                       ("prospective only (generated before 12:00 on the day before)",
                        done[done["before_gate_closure"].astype(str) == "True"])]:
        if sub.empty:
            print(f"\n{label}: none yet")
            continue
        y, p = sub["actual_negative"].astype(int).to_numpy(), sub["probability"].to_numpy()
        both = 0 < y.sum() < len(y)
        r = evaluate(y, (p >= threshold).astype(int), y_score=p if both else None, y_prob=p, name=label)
        print(f"\n{label}: {sub['target_date'].nunique()} days, {len(sub)} hours, {int(y.sum())} negative")
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
               if k in ("precision", "recall", "f1", "pr_auc", "brier")})


# ---------------------------------------------------------------------------
# Section 6: verification against the frozen test run
# ---------------------------------------------------------------------------
def verify(target_date, tol=1e-6):
    """Replay a past day through the serving pipeline and compare with results/final predictions."""
    ref_path = Path("results") / "final" / "predictions_test.csv"
    if not ref_path.exists():
        raise SystemExit("results/final/predictions_test.csv not found: run src/final.py first.")
    ref = pd.read_csv(ref_path, index_col=0)
    ref.index = pd.to_datetime(ref.index, utc=True)
    rec = forecast_day(target_date, source="replay")
    mine = pd.Series({pd.Timestamp(h["hour_utc"]): (np.nan if h["probability"] is None else h["probability"])
                      for h in rec["hours"]})
    theirs = ref["score_LR_rolling_12_months"].reindex(mine.index)
    if theirs.isna().all():
        raise SystemExit("That day is not in the frozen test period (2025-01-01 .. 2026-09-30).")
    same_gaps = bool((mine.isna() == theirs.isna()).all())   # the extra DST hour is absent in both
    both = mine.notna() & theirs.notna()
    diff = (mine[both] - theirs[both]).abs()
    print(f"{target_date}: {int(both.sum())} hours compared, max |difference| = {diff.max():.2e} "
          f"(probabilities rounded to 4 decimals in the output); same hours without forecast: {same_gaps}")
    ok = bool(diff.max() < max(tol, 1.5e-4) and same_gaps)
    print("PASS: serving pipeline reproduces the frozen test predictions." if ok
          else "FAIL: serving pipeline differs from the frozen test run:\n" + diff[diff > tol].to_string())
    return ok


# ---------------------------------------------------------------------------
# Section 7: availability monitor (when does each input actually get published?)
# ---------------------------------------------------------------------------
AVAIL_LOG = OUT_DIR / "availability_log.csv"


def check_availability(target_date):
    """Count published hours of the target day per series and append to outputs/availability_log.csv.

    Run it at several times of day (for example 09:00, 11:00, 13:00, 17:00, 19:00 Berlin time) for a
    few days. The log shows empirically WHEN each input is available, which decides whether a
    'before the auction' forecast is possible.
    """
    target_date = pd.Timestamp(target_date).date().isoformat()
    t_start, t_end = day_bounds(target_date)
    client = _client()
    lo, hi = t_start - pd.Timedelta(days=1), t_end
    counts = {}

    def count(name, fn, col=None):
        try:
            obj = _hourly(_retry(fn))
            ser = obj if isinstance(obj, pd.Series) else (obj[col] if col is not None else obj.iloc[:, 0])
            counts[name] = int(ser.loc[(ser.index >= t_start) & (ser.index < t_end)].notna().sum())
        except Exception as e:  # noqa: BLE001
            counts[name] = 0

    count("DE_price", lambda: client.query_day_ahead_prices("DE_LU", start=lo, end=hi))
    count("DE_load", lambda: client.query_load_forecast("DE_LU", start=lo, end=hi))
    ws_fn = lambda: client.query_wind_and_solar_forecast("DE_LU", start=lo, end=hi)  # noqa: E731
    for col, name in [("Solar", "DE_solar"), ("Wind Onshore", "DE_wind_onshore"), ("Wind Offshore", "DE_wind_offshore")]:
        count(name, ws_fn, col)
    for cc in COUNTRIES:
        df = fetch_country(client, cc, start=lo, end=hi)
        for c in ("load", "solar", "wind"):
            if c in df.columns:
                ser = df[c]
                counts[f"{cc}_{c}"] = int(ser.loc[(ser.index >= t_start) & (ser.index < t_end)].notna().sum())
            else:
                counts[f"{cc}_{c}"] = 0

    now = pd.Timestamp.now(tz="UTC")
    expected = int((t_end - t_start) / pd.Timedelta(hours=1))
    de_names = {"DE_price", "DE_load", "DE_solar", "DE_wind_onshore", "DE_wind_offshore"}
    model_series = de_names | expected_series() if expected_series() else set(counts)   # series the model uses
    rows = pd.DataFrame({"checked_at_utc": now.isoformat(),
                         "checked_at_berlin": now.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M"),
                         "target_date": target_date, "series": list(counts), "hours_published": list(counts.values()),
                         "hours_expected": expected})
    OUT_DIR.mkdir(exist_ok=True)
    rows.to_csv(AVAIL_LOG, mode="a", header=not AVAIL_LOG.exists(), index=False)
    cairo = now.tz_convert("Africa/Cairo").strftime("%H:%M")
    print(f"Published hours for {target_date} (of {expected}) at {rows['checked_at_berlin'].iloc[0]} Berlin time "
          f"({cairo} Cairo):")
    for k, v in counts.items():
        flag = "" if v == expected else ("   <-- incomplete" if k in model_series else "   (not used by the model)")
        print(f"  {k:<18}{v:>3}{flag}")
    print(f"Appended to {AVAIL_LOG}")
    return counts


def print_schedule():
    """Today's check times in Berlin AND Cairo time (daylight-saving aware)."""
    now = pd.Timestamp.now(tz="UTC")
    print(f"Now: {now.tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M} Berlin | {now.tz_convert('Africa/Cairo'):%H:%M} Cairo")
    today = now.tz_convert(LOCAL_TZ).date().isoformat()
    print("\nRun   python src\\predict_tomorrow.py --check-availability   at:")
    for hhmm, why in [("10:30", "KEY reading: is everything out before the 12:00 auction?"),
                      ("13:00", "optional: narrows down when it appears"),
                      ("17:00", "late-afternoon reading"), ("19:00", "confirms completion")]:
        t = pd.Timestamp(f"{today} {hhmm}", tz=LOCAL_TZ)
        print(f"  {hhmm} Berlin  =  {t.tz_convert('Africa/Cairo'):%H:%M} Cairo   ({why})")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="live forecast for this target day (default: tomorrow, Berlin time)")
    ap.add_argument("--replay", help="replay a past day from the local cache (no API, not logged)")
    ap.add_argument("--verify", help="replay a past test-period day and compare with final.py predictions")
    ap.add_argument("--check-availability", action="store_true",
                    help="log how many hours of each input are published for the target day, right now")
    ap.add_argument("--schedule", action="store_true", help="print today's check times in Berlin and Cairo time")
    ap.add_argument("--scenario", help='what-if factors vs yesterday, e.g. "solar=1.3,wind=0.8" (keys: '
                                       'solar, wind, load, nb_solar, nb_wind, nb_load). Not saved, not logged.')
    ap.add_argument("--score", action="store_true", help="fill realised prices into the log and print accuracy")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--refresh", action="store_true", help="ignore the cached live download")
    ap.add_argument("--no-log", action="store_true", help="do not append the live forecast to the log")
    a = ap.parse_args()

    if a.schedule:
        return print_schedule()
    if a.check_availability:
        tomorrow = (pd.Timestamp.now(tz=LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date().isoformat()
        check_availability(a.date or tomorrow)
        return
    if a.score:
        return score_log(a.threshold)
    if a.verify:
        return sys.exit(0 if verify(a.verify) else 1)

    scenario = parse_scenario(a.scenario) if a.scenario else None
    if a.replay:
        record = forecast_day(a.replay, source="replay", threshold=a.threshold, scenario=scenario)
    else:
        target = a.date or (pd.Timestamp.now(tz=LOCAL_TZ).normalize() + pd.Timedelta(days=1)).date().isoformat()
        record = forecast_day(target, source="api", refresh=a.refresh, threshold=a.threshold, scenario=scenario)
        if not a.no_log and not scenario:
            append_log(record)

    path = None if scenario else save_record(record)
    s = record["summary"]
    print(f"\nForecast for {record['target_date']} ({record['source']}) | data status: {record['status']['level']}")
    for n in record["status"]["notes"]:
        print("  note:", n)
    print(f"Expected negative hours: {s['expected_negative_hours']} | highest hourly probability: {s['max_probability']}")
    print("Flagged windows (probability >= %.2f): %s" % (a.threshold, s["flagged_windows"] or "none"))
    print(f"Saved {path}" if path else "(what-if scenario: not saved, not logged)")


if __name__ == "__main__":
    main()