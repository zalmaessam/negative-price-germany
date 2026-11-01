"""
dashboard_logic.py  --  the (Streamlit-free) logic behind app.py, so it can be tested on its own.

Everything here is plain pandas/numpy: threshold maths, hourly tables, windows, scorecards,
track-record metrics, calibration, and the analysis of the availability log.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"
FORECAST_DIR = Path("outputs") / "forecasts"
LOG_FILE = Path("outputs") / "prediction_log.csv"
AVAIL_LOG = Path("outputs") / "availability_log.csv"
TEST_RESULTS = Path("results") / "final" / "test_results.csv"

FEATURE_LABELS = {
    "residual_load": "German residual load (load - wind - solar)",
    "res_share": "German residual share of load",
    "renewable_share": "German wind + solar share of load",
    "load_forecast": "German load forecast",
    "solar_forecast": "German solar forecast",
    "wind_total_forecast": "German wind forecast",
    "price_lag_24": "Price at this hour yesterday",
    "price_lag_168": "Price at this hour last week",
    "neg_count_prev_day": "Negative hours yesterday",
    "price_prev_day_mean": "Average price yesterday",
    "is_weekend": "Weekend",
    "is_holiday": "Public holiday",
    "hour_sin": "Time of day", "hour_cos": "Time of day",
    "month_sin": "Season", "month_cos": "Season",
    "nb_residual_sum": "Neighbours' residual load",
    "nb_load_sum": "Neighbours' load",
    "nb_res_share": "Neighbours' residual share",
    "coupled_residual": "Coupled residual load (Germany + neighbours)",
    "coupled_res_share": "Coupled residual share",
}


# ---------------------------------------------------------------------------
# Threshold and labels
# ---------------------------------------------------------------------------
def cost_ratio_to_threshold(ratio):
    """Flag an hour when p >= 1/(1+ratio), where ratio = (cost of a miss) / (cost of a false alarm).

    This is the cost-optimal cut-off if the probabilities are well calibrated. ratio=1 gives 0.5.
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    return 1.0 / (1.0 + float(ratio))


def pretty_feature(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.endswith("_residual"):
        return f"{name[:-len('_residual')].replace('_', ' ')} residual load"
    return name.replace("_", " ")


def drivers_text(drivers, k=3):
    """'Neighbours' residual load (+1.2), Weekend (-0.4)'. Positive pushes toward a negative price."""
    if not drivers:
        return ""
    return ", ".join(f"{pretty_feature(d['feature'])} ({d['contribution']:+.1f})" for d in drivers[:k])


def label_hours(local_labels, utc_labels):
    """Local hour labels; on a 25-hour day the repeated hour gets a '(2nd)' suffix."""
    seen, out = {}, []
    for lab in local_labels:
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab} (2nd)")
    return out


# ---------------------------------------------------------------------------
# Forecast records
# ---------------------------------------------------------------------------
def list_live_forecasts(directory=None):
    """Saved LIVE forecasts (files forecast_YYYY-MM-DD.json), newest first, as [(date, path)]."""
    d = Path(directory) if directory else FORECAST_DIR
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("forecast_*.json"), reverse=True):
        try:
            with open(f) as fh:
                source = json.load(fh).get("source")
        except Exception:  # noqa: BLE001  (unreadable file: skip it)
            continue
        if source == "api":                      # old replays saved as forecast_*.json are ignored
            out.append((f.stem.replace("forecast_", ""), f))
    return out


def load_record(path):
    with open(path) as fh:
        return json.load(fh)


def hours_frame(record, threshold):
    """Hourly table for display. Flags are recomputed from `threshold` (the slider)."""
    rows = []
    for h in record["hours"]:
        p = h["probability"]
        rows.append({
            "hour_local": h["hour_local"], "hour_utc": h["hour_utc"],
            "probability": np.nan if p is None else float(p),
            "load_mw": h.get("load_mw"), "solar_mw": h.get("solar_mw"), "wind_mw": h.get("wind_mw"),
            "residual_load_mw": h.get("residual_load_mw"), "coupled_residual_mw": h.get("coupled_residual_mw"),
            "drivers": drivers_text(h.get("drivers", [])),
        })
    df = pd.DataFrame(rows)
    df["hour_label"] = label_hours(df["hour_local"].tolist(), df["hour_utc"].tolist())
    df["no_forecast"] = df["probability"].isna()
    df["flag"] = (df["probability"] >= threshold) & ~df["no_forecast"]
    return df


def windows_from_frame(df):
    """Contiguous runs of flagged hours -> [{'start','end','max_probability'}] in local time."""
    out, start = [], None
    flags = df["flag"].tolist()
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if start is not None and (not f or i == len(flags) - 1):
            stop = i if f else i - 1
            end_ts = pd.Timestamp(df["hour_utc"].iloc[stop]).tz_convert(LOCAL_TZ) + pd.Timedelta(hours=1)
            out.append({"start": df["hour_local"].iloc[start], "end": end_ts.strftime("%H:%M"),
                        "max_probability": float(df["probability"].iloc[start:stop + 1].max())})
            start = None
    return out


def summarize(df):
    prob = df["probability"]
    return {
        "expected_negative_hours": float(prob.sum(skipna=True)),
        "max_probability": float(prob.max()) if prob.notna().any() else float("nan"),
        "hours_flagged": int(df["flag"].sum()),
        "hours_without_forecast": int(df["no_forecast"].sum()),
        "windows": windows_from_frame(df),
    }


def attach_actuals(df, prices_utc):
    """Add realised prices (Series indexed by UTC timestamps) and the negative-hour outcome."""
    out = df.copy()
    ts = pd.to_datetime(out["hour_utc"], utc=True)
    out["actual_price"] = ts.map(prices_utc).to_numpy(dtype=float)
    out["actual_negative"] = out["actual_price"] < 0
    return out


def day_scorecard(df):
    """Compare flags with outcomes for one finished day."""
    d = df[~df["no_forecast"] & df["actual_price"].notna()]
    neg, flag = d["actual_negative"].astype(bool), d["flag"].astype(bool)
    tp, fp, fn = int((neg & flag).sum()), int((~neg & flag).sum()), int((neg & ~flag).sum())
    return {
        "negative_hours": int(neg.sum()), "flagged_hours": int(flag.sum()),
        "caught": tp, "false_alarms": fp, "missed": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def negative_hours_by_day(prices_utc, since="2025-01-01", top=5):
    """Days with the most negative-price hours (for the replay examples)."""
    p = prices_utc.dropna()
    local = p.index.tz_convert(LOCAL_TZ)
    day = local.normalize().tz_localize(None)
    s = pd.Series((p.to_numpy() < 0).astype(int), index=day).groupby(level=0).sum()
    s = s[s.index >= pd.Timestamp(since)].sort_values(ascending=False).head(top)
    return [(d.date().isoformat(), int(n)) for d, n in s.items()]


# ---------------------------------------------------------------------------
# Track record (prediction log)
# ---------------------------------------------------------------------------
def load_log(path=None):
    p = Path(path) if path else LOG_FILE
    if not p.exists():
        return None
    log = pd.read_csv(p)
    if log.empty:
        return None
    log["hour_ts"] = pd.to_datetime(log["hour_utc"], utc=True)
    return log


def log_overview(log):
    g = log.groupby("target_date")
    return pd.DataFrame({
        "hours_logged": g.size(),
        "hours_scored": g["actual_price"].apply(lambda s: int(s.notna().sum())),
        "expected_negative_hours": g["probability"].sum(min_count=1).round(1),
        "before_gate_closure": g["before_gate_closure"].first(),
    }).reset_index()


def scored_rows(log, prospective_only=False):
    d = log[log["actual_price"].notna() & log["probability"].notna()]
    if prospective_only:
        d = d[d["before_gate_closure"].astype(str) == "True"]
    return d


def log_metrics(log, threshold, prospective_only=False):
    d = scored_rows(log, prospective_only)
    if d.empty:
        return None
    y = d["actual_negative"].astype(int).to_numpy()
    p = d["probability"].to_numpy(dtype=float)
    pred = (p >= threshold).astype(int)
    tp, fp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"days": int(d["target_date"].nunique()), "hours": int(len(d)), "negative_hours": int(y.sum()),
            "precision": precision, "recall": recall, "f1": f1, "brier": float(np.mean((p - y) ** 2))}


def calibration_table(log, prospective_only=False, bins=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)):
    d = scored_rows(log, prospective_only)
    if d.empty:
        return None
    cut = pd.cut(d["probability"], bins=list(bins), right=False)
    g = d.groupby(cut, observed=True)
    out = pd.DataFrame({"hours": g.size(), "mean_predicted": g["probability"].mean(),
                        "observed_rate": g["actual_negative"].apply(lambda s: float(s.astype(int).mean()))})
    out.index = out.index.astype(str)
    return out


# ---------------------------------------------------------------------------
# Data timing (availability log)
# ---------------------------------------------------------------------------
def load_availability(path=None):
    p = Path(path) if path else AVAIL_LOG
    if not p.exists():
        return None
    av = pd.read_csv(p)
    if av.empty:
        return None
    av["checked_utc"] = pd.to_datetime(av["checked_at_utc"], utc=True, format="ISO8601")
    av["target"] = pd.to_datetime(av["target_date"]).dt.date.astype(str)
    return av


def availability_summary(av):
    """One row per check: which input groups were complete, and how far before/after gate closure."""
    ever = av.groupby("series")["hours_published"].max()
    live = set(ever[ever > 0].index)          # series that have ever been seen (drops never-published ones)
    rows = []
    for (ts, target), g in av.groupby(["checked_utc", "target"]):
        g = g.set_index("series")
        expected = int(g["hours_expected"].iloc[0])

        def complete(names):
            names = [n for n in names if n in g.index]
            return bool(names) and all(g.loc[n, "hours_published"] >= expected for n in names)

        nb = [s for s in g.index if not s.startswith("DE_") and s in live]
        gate = pd.Timestamp((pd.Timestamp(target) - pd.Timedelta(days=1)).date().isoformat() + " 12:00", tz=LOCAL_TZ)
        de_ws, de_load, nb_ok = complete(["DE_solar", "DE_wind_onshore", "DE_wind_offshore"]), complete(["DE_load"]), complete(nb)
        rows.append({
            "checked_berlin": ts.tz_convert(LOCAL_TZ).strftime("%a %Y-%m-%d %H:%M"), "checked_utc": ts, "target_date": target,
            "German_load": de_load, "German_wind_solar": de_ws, "Neighbours": nb_ok,
            "All_inputs_complete": de_load and de_ws and nb_ok,
            "Tomorrows_price_public": complete(["DE_price"]),
            "hours_before_gate_closure": round((gate - ts).total_seconds() / 3600, 1),
        })
    return pd.DataFrame(rows).sort_values("checked_utc").reset_index(drop=True)


def timing_verdict(summary):
    """Plain-language reading of the availability evidence. Returns (headline, [detail lines])."""
    if summary is None or summary.empty:
        return "No availability readings yet.", []
    details = []
    before = summary[summary["hours_before_gate_closure"] > 0]
    for target, g in summary.groupby("target_date"):
        done, todo = g[g["All_inputs_complete"]], g[~g["All_inputs_complete"]]
        first = done.iloc[0] if len(done) else None
        last_inc = todo.iloc[-1] if len(todo) else None
        parts = [f"{target}:"]
        parts.append("first complete reading " + (f"{first['checked_berlin']} Berlin ({abs(first['hours_before_gate_closure']):.1f} h "
                     f"{'before' if first['hours_before_gate_closure'] > 0 else 'after'} gate closure)" if first is not None else "none yet"))
        if last_inc is not None:
            parts.append(f"last incomplete reading {last_inc['checked_berlin']} Berlin")
        details.append(" | ".join(parts))
    if before.empty:
        head = ("No reading before gate closure (12:00 Berlin on the day before) yet. "
                "Take readings around 10:30 Berlin to answer the key question.")
    elif before["All_inputs_complete"].any():
        n_days = before.groupby("target_date")["All_inputs_complete"].any().sum()
        head = (f"All inputs were already complete BEFORE gate closure for {int(n_days)} of "
                f"{before['target_date'].nunique()} target day(s) checked: a before-the-auction forecast looks feasible.")
    else:
        head = ("Inputs were NOT complete before gate closure in any reading so far: a forecast made before the "
                "auction cannot use these inputs.")
    return head, details


# ---------------------------------------------------------------------------
# Validated performance (for the About tab)
# ---------------------------------------------------------------------------
def validated_results(path=None):
    p = Path(path) if path else TEST_RESULTS
    if not p.exists():
        return None
    t = pd.read_csv(p)
    keep = [m for m in ("B3 | static (fit once)", "B3 | rolling 12 months", "B5 | rolling 12 months",
                        "LR | static (fit once)", "LR | rolling 12 months", "XGB | rolling 12 months")
            if m in set(t["model"])]
    return t[t["model"].isin(keep)][["model", "precision", "recall", "f1", "pr_auc", "brier"]].round(3)