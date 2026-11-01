"""
experiments.py  --  four diagnostic experiments on the VALIDATION period (2024).

    python src/experiments.py depth       # how deep are the negatives we catch / miss? (instant)
    python src/experiments.py targets     # does the "price < 0" definition itself cap performance?
    python src/experiments.py features    # day-shape, solar-capacity and neighbour-market features
    python src/experiments.py adaptive    # retraining on recent data (the regime-shift fix)
    python src/experiments.py combined    # adopted ideas together (neighbours + adaptive), vs a fair rule
    python src/experiments.py rules       # is it ML, or just the better input? (rules/small models with neighbours)
    python src/experiments.py all

RULES (put them in the paper):
  * These experiments NEVER touch the test set. They train on 2019-2023 and score 2024.
  * Every experiment is declared in advance and EVERY result is reported, including failures.
  * Hyperparameters are frozen from models.py (results/run_summary_val.json); nothing is
    re-tuned here, so differences come from the idea being tested, not from extra tuning.
  * The sensitivity targets in `targets` are for understanding, not for choosing a prettier
    headline. The primary target stays "hourly mean price < 0".
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from evaluation import evaluate, results_table  # noqa: E402
from features import LINEAR_FEATURES, TREE_FEATURES, fill_neighbour_residual, make_dataset  # noqa: E402
from models import choose_threshold, day_block_bootstrap, make_lr, make_xgb  # noqa: E402
from baselines import fit_residual_threshold  # noqa: E402
from splits import LOCAL_TZ, add_flags, get_split, walk_forward_folds  # noqa: E402

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_PARAMS = {
    "XGB": {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 200},
    "LR": {"C": 0.1},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def best_params(kind):
    """Frozen hyperparameters from the models.py development run (or sensible defaults)."""
    path = RESULTS_DIR / "run_summary_val.json"
    if path.exists():
        return json.load(open(path))[kind]["params"]
    print(f"(run_summary_val.json not found, using default {kind} parameters)")
    return DEFAULT_PARAMS[kind]


def fit_weighted(model, X, y, w=None):
    """Fit with optional sample weights (a Pipeline needs the weight routed to its last step)."""
    if w is None:
        return model.fit(X, y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def oof_threshold_and_final(dev, evl, feats, make_model, params):
    """Walk-forward OOF predictions -> F1-optimal threshold; then fit on all dev, predict evl."""
    parts = []
    for fit_idx, val_idx, _ in walk_forward_folds(dev):
        m = make_model(**params).fit(dev.loc[fit_idx, feats], dev.loc[fit_idx, "target"].astype(int))
        parts.append(pd.Series(m.predict_proba(dev.loc[val_idx, feats])[:, 1], index=val_idx))
    oof = pd.concat(parts)
    thr, _ = choose_threshold(dev.loc[oof.index, "target"], oof)
    final = make_model(**params).fit(dev[feats], dev["target"].astype(int))
    return final.predict_proba(evl[feats])[:, 1], thr, final


# ---------------------------------------------------------------------------
# Extra feature builders (all leakage-safe: day-ahead forecasts only)
# ---------------------------------------------------------------------------
DAY_FEATURES = [
    "res_day_min", "res_day_mean", "res_gap_to_day_min", "res_rank_in_day",
    "res_diff_prev", "res_diff_next", "solar_day_max",
]
SOLAR_FEATURES = ["solar_cf"]


def add_extra_features(data):
    """Within-day shape + capacity-normalised solar.

    * Day-shape features describe where this hour sits in the delivery day's residual-load curve.
      All 24 forecasts of the delivery day are published together, so they are known in advance.
    * res_diff_next looks one hour ahead ONLY inside the same local day, because the forecast
      for the next day is not yet published at gate closure.
    * solar_cf = solar forecast / (highest daily solar forecast over the previous 365 days,
      excluding today). It is a capacity-free 'how sunny relative to the installed fleet'
      measure, so it does not simply count the years.
    """
    full_idx = pd.date_range(data.index.min(), data.index.max(), freq="h", tz="UTC")
    full = data[["residual_load", "solar_forecast"]].reindex(full_idx)
    day = full_idx.tz_convert(LOCAL_TZ).normalize().tz_localize(None)

    res, sol = full["residual_load"], full["solar_forecast"]
    g_res, g_sol = res.groupby(day), sol.groupby(day)

    ex = pd.DataFrame(index=full_idx)
    ex["res_day_min"] = g_res.transform("min")
    ex["res_day_mean"] = g_res.transform("mean")
    ex["res_gap_to_day_min"] = res - ex["res_day_min"]
    ex["res_rank_in_day"] = g_res.rank(pct=True)
    ex["res_diff_prev"] = res.diff(1)
    ex["res_diff_next"] = (g_res.shift(-1) - res).fillna(0.0)
    ex["solar_day_max"] = g_sol.transform("max")

    daily_max = sol.groupby(day).max()
    proxy_daily = daily_max.rolling(365, min_periods=60).max().shift(1)  # strictly before today
    proxy = pd.Series(proxy_daily.reindex(day).to_numpy(), index=full_idx)
    ex["solar_cf"] = sol / proxy.where(proxy > 0)
    return ex.reindex(data.index)


def load_data(with_extra=False):
    data = add_flags(make_dataset())
    nb_cols = []
    if with_extra:
        data = data.join(add_extra_features(data))
        try:
            from neighbors import neighbor_features

            nb, nb_cols = neighbor_features(data.index)
            data = data.join(nb)
            data["coupled_residual"] = data["residual_load"] + data["nb_residual_sum"]
            data["coupled_res_share"] = data["coupled_residual"] / (data["load_forecast"] + data["nb_load_sum"])
            nb_cols = nb_cols + ["coupled_residual", "coupled_res_share"]
        except (FileNotFoundError, RuntimeError) as e:
            print(f"(neighbour features skipped: {e})")
    dev, val = get_split(data, "train"), get_split(data, "val")  # the test set is never loaded here
    return data, dev, val, nb_cols


def _day_ids(index):
    return pd.factorize(index.tz_convert(LOCAL_TZ).normalize())[0]


# ---------------------------------------------------------------------------
# Experiment 1: depth of negative prices (uses predictions saved by models.py)
# ---------------------------------------------------------------------------
def exp_depth():
    path = RESULTS_DIR / "predictions_val.csv"
    if not path.exists():
        raise SystemExit("Run `python src/models.py` first (it saves predictions_val.csv).")
    p = pd.read_csv(path, index_col=0)
    models = {"B3": "pred_B3", "LR": "pred_LR", "XGB": "pred_XGB"}

    neg = p[p["y"] == 1].copy()
    neg["depth"] = pd.cut(neg["price"], [-np.inf, -20, -5, -1, 0],
                          labels=["< -20", "-20..-5", "-5..-1", "-1..0"], right=False)
    g = neg.groupby("depth", observed=False)
    recall = pd.DataFrame({m: g[col].mean() for m, col in models.items()})
    recall.insert(0, "n_hours", g.size())
    print("RECALL BY DEPTH of the negative price (share of those hours each model flags):")
    print(recall.round(3).to_string())

    rows = []
    for m, col in models.items():
        f = p[p[col] == 1]
        rows.append({
            "model": m, "flagged": len(f),
            "precision (price<0)": (f["price"] < 0).mean(),
            "share price<=0": (f["price"] <= 0).mean(),
            "share price<=1": (f["price"] <= 1).mean(),
            "share price<=5": (f["price"] <= 5).mean(),
        })
    tol = pd.DataFrame(rows).set_index("model")
    print("\nPRECISION UNDER TOLERANCE (a flagged hour counts as right if price <= X):")
    print(tol.round(3).to_string())

    miss = {m: p[(p["y"] == 1) & (p[col] == 0)] for m, col in models.items()}
    print("\nMISSED negative hours: how many are barely negative (price > -1)?")
    for m, d in miss.items():
        print(f"  {m}: {len(d)} missed, {(d['price'] > -1).mean():.0%} of them above -1 EUR/MWh")

    recall.to_csv(RESULTS_DIR / "exp_depth_recall.csv")
    tol.to_csv(RESULTS_DIR / "exp_depth_tolerance.csv")


# ---------------------------------------------------------------------------
# Experiment 2: does the target definition cap performance?
# ---------------------------------------------------------------------------
TARGET_DEFS = {
    "price < 0 (primary)": lambda p: p < 0,
    "price <= 0": lambda p: p <= 0,
    "price < 1": lambda p: p < 1,
    "price < -5 (material)": lambda p: p < -5,
}


def exp_targets():
    _, dev, val, _ = load_data()
    params = best_params("XGB")
    rows = []
    for label, fn in TARGET_DEFS.items():
        d, v = dev.assign(target=fn(dev["price"]).astype(int)), val.assign(target=fn(val["price"]).astype(int))
        y = v["target"].to_numpy()
        base = y.mean()
        print(f"[{label}] base rate dev {d['target'].mean():.3%} / val {base:.3%}")

        thr_b3, _ = fit_residual_threshold(d)
        r = evaluate(y, (v["residual_load"] <= thr_b3).astype(int), y_score=-v["residual_load"].to_numpy(),
                     name=f"B3 | {label}")
        r["base_rate"] = base
        rows.append(r)

        p, thr, _ = oof_threshold_and_final(d, v, TREE_FEATURES, make_xgb, params)
        r = evaluate(y, (p >= thr).astype(int), y_score=p, y_prob=p, name=f"XGB | {label}")
        r["base_rate"] = base
        rows.append(r)

    table = results_table(rows)
    table["pr_auc_lift"] = table["pr_auc"] / table["base_rate"]  # comparable across targets
    cols = ["base_rate", "precision", "recall", "f1", "pr_auc", "pr_auc_lift", "brier"]
    print("\nRESULTS BY TARGET DEFINITION (compare lift and B3-vs-XGB gaps, not raw PR-AUC):")
    print(table[cols].round(3).to_string())
    table.to_csv(RESULTS_DIR / "exp_targets.csv")


# ---------------------------------------------------------------------------
# Experiment 3: richer features (day shape, solar capacity, neighbours)
# ---------------------------------------------------------------------------
def exp_features(n_boot=500):
    data, dev, val, nb_cols = load_data(with_extra=True)
    params = best_params("XGB")
    y = val["target"].astype(int).to_numpy()

    sets = {
        "XGB base": TREE_FEATURES,
        "XGB + day shape": TREE_FEATURES + DAY_FEATURES,
        "XGB + solar_cf": TREE_FEATURES + SOLAR_FEATURES,
        "XGB + day shape + solar_cf": TREE_FEATURES + DAY_FEATURES + SOLAR_FEATURES,
    }
    if nb_cols:
        sets["XGB + neighbours"] = TREE_FEATURES + nb_cols
        sets["XGB + everything"] = TREE_FEATURES + DAY_FEATURES + SOLAR_FEATURES + nb_cols

    models, last = {}, None
    thr_b3, _ = fit_residual_threshold(dev)
    models[f"B3 residual load <= {thr_b3:,.0f} MW"] = dict(
        pred=(val["residual_load"] <= thr_b3).astype(int).to_numpy(), score=-val["residual_load"].to_numpy(), prob=None)
    for name, feats in sets.items():
        print(f"fitting {name} ({len(feats)} features)")
        p, thr, last = oof_threshold_and_final(dev, val, feats, make_xgb, params)
        models[name] = dict(pred=(p >= thr).astype(int), score=p, prob=p)
        last = (last, feats)

    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print("\nFEATURE EXPERIMENT (validation 2024):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / "exp_features.csv")

    print(f"\nBootstrapping {n_boot} resamples of days ...")
    ci, diff = day_block_bootstrap(y, models, _day_ids(val.index), n_boot, ref_prefix="XGB base")
    print("\nPaired difference vs 'XGB base' (positive = better):")
    print(diff.round(3).to_string(index=False))
    ci.to_csv(RESULTS_DIR / "exp_features_ci.csv", index=False)
    diff.to_csv(RESULTS_DIR / "exp_features_diff.csv", index=False)

    model, feats = last
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop features of the richest model:")
    print(imp.head(10).round(3).to_string())


# ---------------------------------------------------------------------------
# Experiment 4: adaptive retraining (the regime-shift fix)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "static (fit once)": dict(refit=False, window_days=None, half_life=None),
    "expanding, monthly refit": dict(refit=True, window_days=None, half_life=None),
    "rolling 12 months": dict(refit=True, window_days=365, half_life=None),
    "rolling 24 months": dict(refit=True, window_days=730, half_life=None),
    "expanding + recency weights (half-life 180 d)": dict(refit=True, window_days=None, half_life=180),
}


def rolling(data, evl, strat, fit_fn, out_fn):
    """Walk through the evaluation period month by month.

    For each month, the model is (re)fitted ONLY on labelled hours strictly before that month
    starts (optionally limited to a recent window and/or down-weighted by age), then predicts
    the whole month. This mimics a deployed system that is retrained monthly. Note it does use
    the earlier months of the evaluation year as training data, which is the whole point.
    """
    local = evl.index.tz_convert(LOCAL_TZ)
    ym = (local.year * 100 + local.month).to_numpy()
    first_start = evl.index.min()
    out = pd.Series(np.nan, index=evl.index)
    obj = None
    for m in np.unique(ym):
        rows = evl.index[ym == m]
        if strat["refit"] or obj is None:
            now = rows.min() if strat["refit"] else first_start
            pool = data[data.index < now]
            if strat["window_days"]:
                pool = pool[pool.index >= now - pd.Timedelta(days=strat["window_days"])]
            w = None
            if strat["half_life"]:
                age_days = np.asarray((now - pool.index).total_seconds()) / 86400.0
                w = 0.5 ** (age_days / strat["half_life"])
            obj = fit_fn(pool, w)
        out.loc[rows] = out_fn(obj, evl.loc[rows])
    return out.to_numpy()


def exp_adaptive(n_boot=500):
    data, _, val, _ = load_data()
    y = val["target"].astype(int).to_numpy()
    models = {}

    for kind, make_model, feats in [("XGB", make_xgb, TREE_FEATURES), ("LR", make_lr, LINEAR_FEATURES)]:
        params = best_params(kind)
        for sname, strat in STRATEGIES.items():
            print(f"{kind} | {sname}")
            prob = rolling(
                data, val, strat,
                fit_fn=lambda pool, w: fit_weighted(make_model(**params), pool[feats], pool["target"].astype(int), w),
                out_fn=lambda model, df: model.predict_proba(df[feats])[:, 1],
            )
            # Decision rule fixed at 0.5: equal cost for a miss and a false alarm, no tuning noise
            models[f"{kind} | {sname}"] = dict(pred=(prob >= 0.5).astype(int), score=prob, prob=prob)

    for sname, strat in list(STRATEGIES.items())[:4]:  # B3 has no sample weights
        print(f"B3 | {sname}")
        thr = rolling(data, val, strat, fit_fn=lambda pool, w: fit_residual_threshold(pool)[0],
                      out_fn=lambda t, df: np.full(len(df), t))
        models[f"B3 | {sname}"] = dict(pred=(val["residual_load"].to_numpy() <= thr).astype(int),
                                       score=-val["residual_load"].to_numpy(), prob=None)

    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print("\nADAPTIVE RETRAINING (validation 2024, models decided at probability 0.5):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / "exp_adaptive.csv")

    print(f"\nBootstrapping {n_boot} resamples of days ...")
    ci, diff = day_block_bootstrap(y, models, _day_ids(val.index), n_boot, ref_prefix="XGB | static")
    print("\nPaired difference vs 'XGB | static (fit once)' (positive = better):")
    print(diff.round(3).to_string(index=False))
    ci.to_csv(RESULTS_DIR / "exp_adaptive_ci.csv", index=False)
    diff.to_csv(RESULTS_DIR / "exp_adaptive_diff.csv", index=False)


# ---------------------------------------------------------------------------
# Experiment 5: the adopted ideas together (neighbours + adaptive retraining)
# ---------------------------------------------------------------------------
# Hypothesis H3 (exploratory, report either way): price-LEVEL lags mostly reflect the fuel-price
# regime (2022 averaged ~236 EUR/MWh), not the renewable surplus, so they may transfer badly.
# neg_count_prev_day is kept because it is a count, not a price level.
PRICE_LEVEL_FEATURES = ["price_lag_24", "price_lag_168", "price_prev_day_mean"]
COMBINED_STRATEGIES = ["static (fit once)", "rolling 12 months", "expanding + recency weights (half-life 180 d)"]


def make_lr_imputed(C):
    """LR that tolerates the few missing neighbour values (median fitted on the training pool only)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def exp_combined(n_boot=500):
    data, _, val, nb_cols = load_data(with_extra=True)
    if not nb_cols:
        raise SystemExit("Neighbour features missing. Run `python src/neighbors.py` first.")
    y = val["target"].astype(int).to_numpy()
    models = {}

    for sname in ("static (fit once)", "rolling 12 months"):  # fair rule baselines
        thr = rolling(data, val, STRATEGIES[sname], fit_fn=lambda pool, w: fit_residual_threshold(pool)[0],
                      out_fn=lambda t, df: np.full(len(df), t))
        models[f"B3 | {sname}"] = dict(pred=(val["residual_load"].to_numpy() <= thr).astype(int),
                                       score=-val["residual_load"].to_numpy(), prob=None)

    plan = [
        ("XGB", make_xgb, TREE_FEATURES),
        ("LR", make_lr_imputed, LINEAR_FEATURES),
    ]
    for kind, make_model, base in plan:
        params = best_params(kind)
        fsets = {
            "base": base,
            "+neighbours": base + nb_cols,
            "+neighbours, no price levels": [f for f in base if f not in PRICE_LEVEL_FEATURES] + nb_cols,
        }
        for fname, feats in fsets.items():
            for sname in COMBINED_STRATEGIES:
                print(f"{kind} | {fname} | {sname}")
                prob = rolling(
                    data, val, STRATEGIES[sname],
                    fit_fn=lambda pool, w: fit_weighted(make_model(**params), pool[feats], pool["target"].astype(int), w),
                    out_fn=lambda model, df: model.predict_proba(df[feats])[:, 1],
                )
                models[f"{kind} | {fname} | {sname}"] = dict(pred=(prob >= 0.5).astype(int), score=prob, prob=prob)

    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print("\nCOMBINED EXPERIMENT (validation 2024, models decided at probability 0.5):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / "exp_combined.csv")

    print(f"\nBootstrapping {n_boot} resamples of days ...")
    ci, diff = day_block_bootstrap(y, models, _day_ids(val.index), n_boot, ref_prefix="B3 | rolling 12 months")
    print("\nPaired difference vs the FAIR RULE 'B3 | rolling 12 months' (positive = better than the rule):")
    print(diff.round(3).to_string(index=False))
    ci.to_csv(RESULTS_DIR / "exp_combined_ci.csv", index=False)
    diff.to_csv(RESULTS_DIR / "exp_combined_diff.csv", index=False)


# ---------------------------------------------------------------------------
# Experiment 6: is it the ML, or just the neighbour information?
# ---------------------------------------------------------------------------
def exp_rules(n_boot=500):
    """Give the SIMPLE methods the neighbour information too, then compare with the full models.

    If a one-line rule on 'German + neighbour residual load' already matches the ML models, the
    honest conclusion is that the coupled-market input is what matters, not the algorithm.
    """
    data, _, val, nb_cols = load_data(with_extra=True)
    if not nb_cols:
        raise SystemExit("Neighbour features missing. Run `python src/neighbors.py` first.")
    data, n_missing = fill_neighbour_residual(data)
    print(f"Hours with a missing neighbour residual (filled forward from the past only): {n_missing}")
    val = data.loc[val.index]
    y = val["target"].astype(int).to_numpy()
    strat = STRATEGIES["rolling 12 months"]
    models = {}

    def add_rule(name, col):
        thr = rolling(data, val, strat, fit_fn=lambda pool, w: fit_residual_threshold(pool, col)[0],
                      out_fn=lambda t, df: np.full(len(df), t))
        models[name] = dict(pred=(val[col].to_numpy() <= thr).astype(int), score=-val[col].to_numpy(), prob=None)

    def add_model(name, make_model, params, feats):
        prob = rolling(data, val, strat,
                       fit_fn=lambda pool, w: fit_weighted(make_model(**params), pool[feats], pool["target"].astype(int), w),
                       out_fn=lambda model, df: model.predict_proba(df[feats])[:, 1])
        models[name] = dict(pred=(prob >= 0.5).astype(int), score=prob, prob=prob)

    add_rule("B3 | German residual-load rule (rolling 12 m)", "residual_load")
    add_rule("B5 | coupled residual-load rule (rolling 12 m)", "coupled_residual")
    cal = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"]
    lr_params = best_params("LR")
    add_model("B6 | LR on German + neighbour residual (rolling 12 m)", make_lr_imputed, lr_params,
              ["residual_load", "nb_residual_sum"])
    add_model("B7 | LR on both residuals + calendar (rolling 12 m)", make_lr_imputed, lr_params,
              ["residual_load", "nb_residual_sum"] + cal)
    add_model("LR | +neighbours, all features (rolling 12 m)", make_lr_imputed, lr_params, LINEAR_FEATURES + nb_cols)
    add_model("XGB | +neighbours, all features (rolling 12 m)", make_xgb, best_params("XGB"), TREE_FEATURES + nb_cols)

    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print("\nRULES AND SMALL MODELS vs FULL MODELS (validation 2024, rolling 12 months):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / "exp_rules.csv")

    print(f"\nBootstrapping {n_boot} resamples of days ...")
    ci, diff = day_block_bootstrap(y, models, _day_ids(val.index), n_boot, ref_prefix="B5 |")
    print("\nPaired difference vs 'B5 | coupled residual-load rule' (positive = better than that rule):")
    print(diff.round(3).to_string(index=False))
    ci.to_csv(RESULTS_DIR / "exp_rules_ci.csv", index=False)
    diff.to_csv(RESULTS_DIR / "exp_rules_diff.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["depth", "targets", "features", "adaptive", "combined", "rules", "all"])
    parser.add_argument("--boots", type=int, default=500)
    a = parser.parse_args()
    todo = ["depth", "targets", "features", "adaptive"] if a.which == "all" else [a.which]
    for name in todo:
        print(f"\n{'=' * 70}\nEXPERIMENT: {name}\n{'=' * 70}")
        {"depth": exp_depth, "targets": exp_targets,
         "features": lambda: exp_features(a.boots), "adaptive": lambda: exp_adaptive(a.boots),
         "combined": lambda: exp_combined(a.boots), "rules": lambda: exp_rules(a.boots)}[name]()