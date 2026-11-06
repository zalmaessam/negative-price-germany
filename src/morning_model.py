"""
morning_model.py  --  how much skill survives if we only use information available BEFORE the auction?

    python src/morning_model.py                # development: scores 2024 (validation). Safe to repeat.
    python src/morning_model.py --test         # scores the test period ONCE (exploratory, see below)
    python src/morning_model.py --boots 2000

Information sets (all known by about 10:30 Berlin time on the day before delivery):
    M0  calendar only                         (hour, season, weekend, holiday)
    M1  + price history                       (same hour yesterday / last week, yesterday's negatives and mean)
    M2  + yesterday's published forecasts     (weather persistence: what the market already expected for D-1)
    M3  + tomorrow's LOAD forecast            (usually published earlier than wind/solar; check the availability log)
    M4  price history + tomorrow's load only  (added AFTER seeing that yesterday's forecasts (M2) added nothing:
                                               a post-hoc simplification, report it as such)
Reference (NOT available in the morning): FULL = the validated model that uses tomorrow's wind/solar forecasts
for Germany and its neighbours.

Besides the hourly scores, a DAY-level score is reported: "will this day have at least 1 (or at least 3)
negative hours?", scored with the expected number of negative hours (sum of the hourly probabilities).

Rules: same rolling-12-month logistic regression as the main study, probability 0.5 decision, rows with
incomplete inputs dropped for every model alike, day-level bootstrap.
The test period was already used for the main study, so a --test run here is EXPLORATORY: report it as
such, run it once, and do not tune anything afterwards (a guard file enforces the single run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parent))
from evaluation import evaluate, results_table  # noqa: E402
from experiments import (  # noqa: E402
    STRATEGIES, _day_ids, best_params, fill_neighbour_residual, fit_weighted, load_data, make_lr_imputed, rolling,
)
from features import (  # noqa: E402
    LINEAR_FEATURES, MORNING_CAL, MORNING_HIST, MORNING_LOAD, MORNING_PERSIST, add_morning_features,
)
from models import day_block_bootstrap  # noqa: E402
from splits import get_split  # noqa: E402

OUT = Path("results") / "morning"
SETS = {
    "M0 calendar only": MORNING_CAL,
    "M1 + price history": MORNING_CAL + MORNING_HIST,
    "M2 + yesterday's published forecasts": MORNING_CAL + MORNING_HIST + MORNING_PERSIST,
    "M3 + tomorrow's load forecast": MORNING_CAL + MORNING_HIST + MORNING_PERSIST + MORNING_LOAD,
    "M4 price history + tomorrow's load (no yesterday's forecasts)": MORNING_CAL + MORNING_HIST + MORNING_LOAD,
}
ALL_MORNING = sorted(set(sum(SETS.values(), [])))
STRAT = STRATEGIES["rolling 12 months"]


DAY_KS = (1, 3)   # a "negative day" has at least 1 / at least 3 negative hours


def day_level(evl, models, n_boot, seed=42):
    """Day-level skill. Score of a day = expected (or, for the naive rules, predicted) number of negative hours.
    A day is flagged when that score >= k. Intervals resample whole days; differences are paired vs B1."""
    day = _day_ids(evl.index)
    y = evl["target"].astype(int).to_numpy()
    n_neg = np.bincount(day, weights=y).astype(int)
    n_days = len(n_neg)
    scores = {n: np.bincount(day, weights=(m["prob"] if m["prob"] is not None else m["pred"].astype(float)))
              for n, m in models.items()}
    draws = np.random.default_rng(seed).integers(0, n_days, size=(n_boot, n_days))
    rows = []
    for k in DAY_KS:
        yk = (n_neg >= k).astype(int)
        if yk.sum() == 0 or yk.sum() == n_days:
            continue
        boot = {}
        for n, sc in scores.items():
            boot[n] = np.array([average_precision_score(yk[d], sc[d]) for d in draws if yk[d].sum() > 0])
        ref = next(n for n in scores if n.startswith("B1"))
        for n, sc in scores.items():
            flag = sc >= k
            tp, fp, fn = int((flag & (yk == 1)).sum()), int((flag & (yk == 0)).sum()), int((~flag & (yk == 1)).sum())
            diff = boot[n] - boot[ref]
            rows.append({
                "model": n, "days_with_k_or_more_negative_hours": k, "days": n_days, "such_days": int(yk.sum()),
                "base_rate": yk.mean(), "pr_auc": average_precision_score(yk, sc),
                "pr_auc_low": np.percentile(boot[n], 2.5), "pr_auc_high": np.percentile(boot[n], 97.5),
                "diff_vs_B1": diff.mean(), "diff_low": np.percentile(diff, 2.5), "diff_high": np.percentile(diff, 97.5),
                "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
            })
    return pd.DataFrame(rows)


def run(eval_name="val", n_boot=1000):
    t0 = time.time()
    data, _, _, nb_cols = load_data(with_extra=True)
    if nb_cols:
        data, _ = fill_neighbour_residual(data)
    data = data.join(add_morning_features(data))
    evl = get_split(data, eval_name, allow_test=(eval_name == "test"))
    n0 = len(evl)
    evl = evl.dropna(subset=ALL_MORNING)           # identical rows for every model
    print(f"Evaluation period ({eval_name}): {evl.index.min().date()} .. {evl.index.max().date()} | "
          f"{len(evl)} hours ({n0 - len(evl)} dropped for incomplete inputs) | {int(evl['target'].sum())} negative")
    y = evl["target"].astype(int).to_numpy()
    params = best_params("LR")

    models = {
        "B1 same hour yesterday": dict(pred=(evl["price_lag_24"] < 0).astype(int).to_numpy(),
                                       score=-evl["price_lag_24"].to_numpy(), prob=None),
        "B2 same hour last week": dict(pred=(evl["price_lag_168"] < 0).astype(int).to_numpy(),
                                       score=-evl["price_lag_168"].to_numpy(), prob=None),
    }

    def add(name, feats):
        print(f"fitting {name} ({len(feats)} features)")
        prob = rolling(data, evl, STRAT,
                       fit_fn=lambda pool, w: fit_weighted(make_lr_imputed(**params), pool[feats],
                                                           pool["target"].astype(int), w),
                       out_fn=lambda model, df: model.predict_proba(df[feats])[:, 1])
        models[name] = dict(pred=(prob >= 0.5).astype(int), score=prob, prob=prob)

    for name, feats in SETS.items():
        add(name, feats)
    if nb_cols:
        add("FULL (needs evening forecasts, NOT available in the morning)", LINEAR_FEATURES + nb_cols)

    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print(f"\n=== MORNING-INFORMATION MODELS ({eval_name.upper()}) ===")
    print(table.round(3).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / f"morning_{eval_name}.csv")

    print(f"\nBootstrapping {n_boot} resamples of days ...")
    refs = ["M0", "M1", "M2", "M3", "M4", "B1"]
    ci, diff = day_block_bootstrap(y, models, _day_ids(evl.index), n_boot, ref_prefix=refs)
    ci.to_csv(OUT / f"morning_ci_{eval_name}.csv", index=False)
    diff.to_csv(OUT / f"morning_diff_{eval_name}.csv", index=False)
    pairs = [
        ("M1 + price history", "M0"), ("M2 + yesterday's published forecasts", "M1"),
        ("M3 + tomorrow's load forecast", "M2"), ("M2 + yesterday's published forecasts", "M0"),
        ("M2 + yesterday's published forecasts", "B1"),
        ("M4 price history + tomorrow's load (no yesterday's forecasts)", "M1"),
        ("M4 price history + tomorrow's load (no yesterday's forecasts)", "M3"),
    ]
    if nb_cols:
        pairs.append(("FULL (needs evening forecasts, NOT available in the morning)", "M2"))
    rows = []
    for a, b in pairs:
        sel = diff[(diff["model"] == a) & (diff["vs"].str.startswith(b))]
        rows.append(sel)
    key = pd.concat(rows, ignore_index=True)
    print("\n=== PAIRED DIFFERENCES (positive = first model better) ===")
    print(key.drop(columns=["share_draws_better"]).round(3).to_string(index=False))
    key.to_csv(OUT / f"morning_key_{eval_name}.csv", index=False)

    dayt = day_level(evl, models, n_boot)
    dayt.to_csv(OUT / f"morning_day_level_{eval_name}.csv", index=False)
    print("\n=== DAY LEVEL: will the day have at least k negative hours? (score = expected number of negative hours) ===")
    if dayt.empty:
        print("(not computed: every day, or no day, reaches the threshold in this period)")
    else:
        show = dayt.copy()
        show["model"] = show["model"].str.slice(0, 44)
        print(show.drop(columns=["days"]).round(3).to_string(index=False))

    if eval_name == "test":
        json.dump({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "n_hours": int(len(evl)), "n_negative": int(y.sum()),
                   "note": "exploratory: the test period was already used for the main study"},
                  open(OUT / "TEST_DONE.json", "w"), indent=2)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} minutes. Saved to {OUT}.")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--boots", type=int, default=1000)
    a = ap.parse_args()
    if a.test and (OUT / "TEST_DONE.json").exists() and not a.rerun:
        raise SystemExit("The exploratory test run was already done (results/morning/TEST_DONE.json). "
                         "Use --rerun only if you will say so in the paper.")
    run("test" if a.test else "val", n_boot=a.boots)