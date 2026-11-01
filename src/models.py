"""
models.py  --  logistic regression and XGBoost vs the baselines, with confidence intervals.

Usage (from the repo root):
    python src/models.py                 # development run: fit on train, score on VALIDATION (2024)
    python src/models.py --final         # final run: fit on train+val, score on TEST. Run ONCE.
    python src/models.py --boots 2000    # more bootstrap resamples (default 1000)

Design decisions (put these in the paper's methods section):
  1. DEVELOPMENT DATA. In a development run the models learn from 2019-2023 only. In the final run
     they learn from 2019-2024. Hyperparameters, thresholds and the B3 cut-off are all chosen
     inside the development data. The evaluation split is never used to choose anything.
  2. TUNING by walk-forward cross-validation (expanding window, one calendar year per fold),
     scored by PR-AUC (average precision), which suits rare positives.
  3. NO class re-weighting. Re-weighting distorts probabilities and wrecks the Brier score.
     Instead we keep honest probabilities and pick the DECISION THRESHOLD that maximises F1 on
     the out-of-fold predictions.
  4. CONFIDENCE INTERVALS by bootstrap that resamples whole delivery DAYS (hours within a day
     are strongly correlated, so resampling single hours would give intervals that are too narrow).
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from baselines import fit_residual_threshold, predict_b1, predict_b2, predict_b3  # noqa: E402
from evaluation import base_rate_row, evaluate, results_table  # noqa: E402
from features import LINEAR_FEATURES, TREE_FEATURES, make_dataset  # noqa: E402
from splits import LOCAL_TZ, add_flags, get_split, walk_forward_folds  # noqa: E402

SEED = 42
RESULTS_DIR = Path("results")
MODEL_DIR = RESULTS_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Section 1: models and (deliberately small) hyperparameter grids
# ---------------------------------------------------------------------------
LR_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}   # C = inverse regularisation strength
XGB_GRID = {                               # 2 x 2 x 2 = 8 combinations
    "max_depth": [3, 5],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [200, 500],
}


def make_lr(C):
    """Scale features (mean 0, sd 1) then fit logistic regression. Scaling matters for LR."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def make_xgb(max_depth, learning_rate, n_estimators):
    """Gradient-boosted trees. Subsampling and min_child_weight are fixed to limit overfitting."""
    return XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=SEED,
        n_jobs=-1,
    )


def _py(x):
    """numpy scalar -> plain Python number (safe to pass back into model constructors)."""
    return x.item() if hasattr(x, "item") else x


# ---------------------------------------------------------------------------
# Section 2: tuning by walk-forward cross-validation
# ---------------------------------------------------------------------------
def tune(make_model, grid, features, dev, label):
    """Try every grid combination on the walk-forward folds.

    For each combination and each fold: fit on the past, predict the next year, score PR-AUC.
    The best combination has the highest MEAN PR-AUC across folds. We also keep every
    out-of-fold (OOF) prediction, which is used next to choose the decision threshold.
    """
    keys = list(grid)
    rows, oof_store = [], {}
    for values in product(*grid.values()):
        params = dict(zip(keys, values))
        parts, fold_ap = [], {}
        for fit_idx, val_idx, year in walk_forward_folds(dev):
            model = make_model(**params)
            model.fit(dev.loc[fit_idx, features], dev.loc[fit_idx, "target"].astype(int))
            p = model.predict_proba(dev.loc[val_idx, features])[:, 1]
            parts.append(pd.Series(p, index=val_idx))
            fold_ap[year] = average_precision_score(dev.loc[val_idx, "target"].astype(int), p)
        oof_store[tuple(values)] = pd.concat(parts)
        rows.append({**params, **{f"AP_{y}": a for y, a in fold_ap.items()},
                     "mean_AP": float(np.mean(list(fold_ap.values())))})
        print(f"  [{label}] {params} -> mean PR-AUC {rows[-1]['mean_AP']:.3f}")

    cv = pd.DataFrame(rows).sort_values("mean_AP", ascending=False).reset_index(drop=True)
    best = {k: _py(cv.loc[0, k]) for k in keys}
    return best, cv, oof_store[tuple(best.values())]


def choose_threshold(y_true, prob):
    """Probability cut-off that maximises F1 on out-of-fold predictions (training data only)."""
    p, r, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(prob))
    p, r = p[:-1], r[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    return float(thr[best]), float(f1[best])


# ---------------------------------------------------------------------------
# Section 3: bootstrap confidence intervals (resampling whole days)
# ---------------------------------------------------------------------------
def _f1(y, pred):
    tp = np.sum(y * pred)
    denom = 2 * tp + np.sum((1 - y) * pred) + np.sum(y * (1 - pred))
    return 2 * tp / denom if denom > 0 else 0.0


def day_block_bootstrap(y, models, day_ids, n_boot, seed=SEED, ref_prefix="B3"):
    """Resample delivery days with replacement and recompute metrics each time.

    models: {name: dict(pred=0/1 array, score=ranking array, prob=probabilities or None)}
    The SAME resampled days are used for every model, so differences between models
    (paired differences) are measured fairly.
    Paired differences are taken against the first model whose name starts with ref_prefix
    (a string, or a list of strings for several references).
    Returns (ci_table, paired_difference_table).
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(day_ids, kind="stable")
    cuts = np.flatnonzero(np.diff(day_ids[order])) + 1
    groups = np.split(order, cuts)  # one array of row positions per day
    n_days = len(groups)

    names = list(models)
    draws = {n: {"f1": [], "pr_auc": [], "brier": []} for n in names}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, n_days, n_days)])
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for n in names:
            m = models[n]
            draws[n]["f1"].append(_f1(yb, m["pred"][idx]))
            draws[n]["pr_auc"].append(average_precision_score(yb, m["score"][idx]))
            if m["prob"] is not None:
                draws[n]["brier"].append(np.mean((m["prob"][idx] - yb) ** 2))

    ci_rows = []
    for n in names:
        m = models[n]
        point = {"f1": _f1(y, m["pred"]), "pr_auc": average_precision_score(y, m["score"]),
                 "brier": np.mean((m["prob"] - y) ** 2) if m["prob"] is not None else np.nan}
        for metric, est in point.items():
            d = np.array(draws[n][metric])
            lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan))
            ci_rows.append({"model": n, "metric": metric, "estimate": est, "ci_low": lo, "ci_high": hi})
    ci = pd.DataFrame(ci_rows)

    prefixes = [ref_prefix] if isinstance(ref_prefix, str) else list(ref_prefix)
    diff_rows = []
    for prefix in prefixes:
        ref = next(n for n in names if n.startswith(prefix))
        for n in names:
            if n == ref:
                continue
            for metric in ("f1", "pr_auc"):
                d = np.array(draws[n][metric]) - np.array(draws[ref][metric])
                diff_rows.append({
                    "model": n, "metric": metric, "vs": ref,
                    "mean_diff": d.mean(), "ci_low": np.percentile(d, 2.5),
                    "ci_high": np.percentile(d, 97.5), "share_draws_better": float((d > 0).mean()),
                })
    return ci, pd.DataFrame(diff_rows)


# ---------------------------------------------------------------------------
# Section 4: the full experiment
# ---------------------------------------------------------------------------
def run(eval_name="val", n_boot=1000):
    data = add_flags(make_dataset())
    train, val = get_split(data, "train"), get_split(data, "val")
    if eval_name == "test":
        dev = pd.concat([train, val])
        evl = get_split(data, "test", allow_test=True)
    else:
        dev, evl = train, val
    print(f"Development data: {dev.index.min().date()} .. {dev.index.max().date()} ({len(dev)} rows)")
    print(f"Evaluation data ({eval_name}): {evl.index.min().date()} .. {evl.index.max().date()} ({len(evl)} rows)\n")

    y_dev = dev["target"].astype(int)
    y = evl["target"].astype(int).to_numpy()

    fitted = {}
    for label, make_model, grid, feats in [
        ("LR", make_lr, LR_GRID, LINEAR_FEATURES),
        ("XGB", make_xgb, XGB_GRID, TREE_FEATURES),
    ]:
        print(f"== Tuning {label} with walk-forward CV ==")
        best, cv, oof = tune(make_model, grid, feats, dev, label)
        cv.to_csv(RESULTS_DIR / f"cv_{label.lower()}_{eval_name}.csv", index=False)
        thr, oof_f1 = choose_threshold(y_dev.loc[oof.index], oof)
        print(f"  best params: {best} | threshold {thr:.3f} (OOF F1 {oof_f1:.3f})\n")

        model = make_model(**best).fit(dev[feats], y_dev)  # refit on ALL development data
        fitted[label] = {"model": model, "features": feats, "params": best, "threshold": thr,
                         "prob": model.predict_proba(evl[feats])[:, 1]}
        joblib.dump({k: fitted[label][k] for k in ("model", "features", "params", "threshold")},
                    MODEL_DIR / f"{label.lower()}_{eval_name}.joblib")

    # Baselines, fitted on the same development data
    thr_b3, _ = fit_residual_threshold(dev)
    b1_pred, b1_score = predict_b1(evl)
    b2_pred, b2_score = predict_b2(evl)
    b3_pred, b3_score = predict_b3(evl, thr_b3)
    b3_name = f"B3 residual load <= {thr_b3:,.0f} MW"

    models = {
        "B1 same hour yesterday": dict(pred=b1_pred.to_numpy(), score=b1_score.to_numpy(), prob=None),
        "B2 same hour last week": dict(pred=b2_pred.to_numpy(), score=b2_score.to_numpy(), prob=None),
        b3_name: dict(pred=b3_pred.to_numpy(), score=b3_score.to_numpy(), prob=None),
    }
    for label, long_name in [("LR", "Logistic regression"), ("XGB", "XGBoost")]:
        f = fitted[label]
        models[long_name] = dict(pred=(f["prob"] >= f["threshold"]).astype(int),
                                 score=f["prob"], prob=f["prob"])

    # Point estimates
    rows = [evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()]
    rows.append(base_rate_row(y))
    table = results_table(rows)
    print(f"== Results on {eval_name.upper()} ==")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / f"models_{eval_name}.csv")

    # Save predictions for the later analyses (year-by-year, SHAP, error analysis, costs)
    pred_df = pd.DataFrame({"y": y, "price": evl["price"].to_numpy(),
                            "flag_2022": evl["flag_2022"].to_numpy(),
                            "flag_15min_era": evl["flag_15min_era"].to_numpy()}, index=evl.index)
    for n, m in models.items():
        short = n.split()[0] if n.startswith("B") else ("LR" if "Logistic" in n else "XGB")
        pred_df[f"pred_{short}"] = m["pred"]
        pred_df[f"score_{short}"] = m["score"]
    pred_df.to_csv(RESULTS_DIR / f"predictions_{eval_name}.csv")

    # Bootstrap confidence intervals
    print(f"\nBootstrapping {n_boot} resamples of whole days (this can take a minute)...")
    day_ids = pd.factorize(evl.index.tz_convert(LOCAL_TZ).normalize())[0]
    ci, diff = day_block_bootstrap(y, models, day_ids, n_boot)
    ci.to_csv(RESULTS_DIR / f"models_ci_{eval_name}.csv", index=False)
    diff.to_csv(RESULTS_DIR / f"models_diff_vs_b3_{eval_name}.csv", index=False)

    show = ci.assign(text=lambda d: d.apply(
        lambda r: "n/a" if np.isnan(r["estimate"]) else f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1))
    print("\n95% confidence intervals (estimate [low, high]):")
    print(show.pivot(index="model", columns="metric", values="text").loc[list(models)].to_string())
    print("\nPaired difference vs B3 (positive = better than B3):")
    print(diff.round(3).to_string(index=False))

    # Quick look at what each model uses
    xgb = fitted["XGB"]
    imp = pd.Series(xgb["model"].feature_importances_, index=xgb["features"]).sort_values(ascending=False)
    print("\nXGBoost top features (gain importance):")
    print(imp.head(8).round(3).to_string())
    lr = fitted["LR"]
    coef = pd.Series(lr["model"][-1].coef_[0], index=lr["features"])
    print("\nLogistic regression largest standardised coefficients:")
    print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(8).round(2).to_string())

    with open(RESULTS_DIR / f"run_summary_{eval_name}.json", "w") as fh:
        json.dump({"eval": eval_name, "b3_threshold_mw": thr_b3,
                   "LR": {"params": fitted["LR"]["params"], "threshold": fitted["LR"]["threshold"]},
                   "XGB": {"params": fitted["XGB"]["params"], "threshold": fitted["XGB"]["threshold"]}},
                  fh, indent=2)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="fit on train+val, score the TEST set (run once)")
    parser.add_argument("--boots", type=int, default=1000)
    args = parser.parse_args()
    run("test" if args.final else "val", n_boot=args.boots)