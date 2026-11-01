"""
final.py  --  the ONE-SHOT test-set evaluation with the configuration frozen before testing.

    python src/final.py                # run once (about 15-30 minutes)
    python src/final.py --boots 2000   # more bootstrap resamples
    python src/final.py --rerun        # overrides the run-once guard; breaks the protocol, say so in the paper

FROZEN PROTOCOL (declared before any test data was scored):
  Data       2019-01 .. 2026-09. Test = 2025-01-01 .. 2026-09-30 (local). The 2-hour partial day
             of 1 Oct 2026 is excluded. Development data = everything before each prediction month.
  Features   base features + neighbour features (9 countries, Poland without solar). Price lags kept.
  Models     Logistic regression and XGBoost, hyperparameters from the development run
             (results/run_summary_val.json), NOT re-tuned.
  Training   PRIMARY: rolling 12 months, refit monthly.  SECONDARY: expanding window with recency
             weights (half-life 180 days).  REFERENCE: static fit once on all data before 2025.
  Decision   probability >= 0.5 (equal cost for a miss and a false alarm).
  Rules      B3 (German residual load), B5 (German + neighbour residual load), each static and
             rolling 12 months; B6/B7 small logistic regressions as secondary references.
  Pre-declared expectations (from validation 2024):
    H1  LR/XGB (primary) beat B3 rolling on PR-AUC and F1.
    H2  LR/XGB (primary) are NOT significantly better than B5 rolling.
    H3  Rolling retraining beats static fitting for LR, XGB and the rules.
  Everything is reported, whatever the outcome.
"""
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from baselines import fit_residual_threshold  # noqa: E402
from evaluation import evaluate, results_table  # noqa: E402
from experiments import (  # noqa: E402
    STRATEGIES, _day_ids, best_params, fill_neighbour_residual, fit_weighted,
    load_data, make_lr_imputed, rolling,
)
from features import LINEAR_FEATURES, TREE_FEATURES  # noqa: E402
from models import day_block_bootstrap, make_xgb  # noqa: E402
from splits import LOCAL_TZ, get_split  # noqa: E402

OUT = Path("results") / "final"
FIG = OUT / "figures"
MARK = OUT / "RUN_COMPLETE.json"

STATIC = "static (fit once)"
PRIMARY = "rolling 12 months"
SECONDARY = "expanding + recency weights (half-life 180 d)"
DECISION_THRESHOLD = 0.5
LAST_LOCAL_DAY = pd.Timestamp("2026-10-01", tz=LOCAL_TZ)  # exclude the 2-hour partial day
CAL = ["hour_sin", "hour_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"]


# ---------------------------------------------------------------------------
# Section 1: building every compared method on the test period
# ---------------------------------------------------------------------------
def rule_model(data, test, col, strat):
    """Threshold rule on `col`, threshold refitted per the strategy (F1-optimal on past data)."""
    thr = rolling(data, test, strat, fit_fn=lambda pool, w: fit_residual_threshold(pool, col)[0],
                  out_fn=lambda t, df: np.full(len(df), t))
    return dict(pred=(test[col].to_numpy() <= thr).astype(int), score=-test[col].to_numpy(), prob=None)


def prob_model(data, test, make_model, params, feats, strat):
    """Probabilistic model retrained per the strategy, decided at the fixed 0.5 cut-off."""
    prob = rolling(data, test, strat,
                   fit_fn=lambda pool, w: fit_weighted(make_model(**params), pool[feats], pool["target"].astype(int), w),
                   out_fn=lambda model, df: model.predict_proba(df[feats])[:, 1])
    return dict(pred=(prob >= DECISION_THRESHOLD).astype(int), score=prob, prob=prob)


def build_models(data, test, nb_cols):
    models = {}
    for sname in (STATIC, PRIMARY):
        print(f"B3 | {sname}")
        models[f"B3 | {sname}"] = rule_model(data, test, "residual_load", STRATEGIES[sname])
        print(f"B5 | {sname}")
        models[f"B5 | {sname}"] = rule_model(data, test, "coupled_residual", STRATEGIES[sname])

    lr_p, xgb_p = best_params("LR"), best_params("XGB")
    print("B6 | LR on two residuals | rolling 12 months")
    models["B6 | LR on two residuals | rolling 12 months"] = prob_model(
        data, test, make_lr_imputed, lr_p, ["residual_load", "nb_residual_sum"], STRATEGIES[PRIMARY])
    print("B7 | LR on two residuals + calendar | rolling 12 months")
    models["B7 | LR on two residuals + calendar | rolling 12 months"] = prob_model(
        data, test, make_lr_imputed, lr_p, ["residual_load", "nb_residual_sum"] + CAL, STRATEGIES[PRIMARY])

    for kind, make_model, params, feats in [
        ("LR", make_lr_imputed, lr_p, LINEAR_FEATURES + nb_cols),
        ("XGB", make_xgb, xgb_p, TREE_FEATURES + nb_cols),
    ]:
        for sname in (STATIC, PRIMARY, SECONDARY):
            print(f"{kind} | {sname}")
            models[f"{kind} | {sname}"] = prob_model(data, test, make_model, params, feats, STRATEGIES[sname])
    return models


# ---------------------------------------------------------------------------
# Section 2: slicing the results (era, quarter, depth, data completeness)
# ---------------------------------------------------------------------------
def subset_table(models, y, mask, label):
    rows = []
    for n, m in models.items():
        if y[mask].sum() == 0:
            continue
        r = evaluate(y[mask], m["pred"][mask], y_score=m["score"][mask],
                     y_prob=None if m["prob"] is None else m["prob"][mask], name=n)
        r["subset"] = label
        rows.append(r)
    return pd.DataFrame(rows)


def by_quarter(models, y, test):
    loc = test.index.tz_convert(LOCAL_TZ)
    q = np.asarray(loc.year.astype(str) + "Q" + loc.quarter.astype(str))
    rows = []
    for quarter in sorted(set(q)):
        t = subset_table(models, y, q == quarter, quarter)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def depth_tables(models, test, keys):
    price = test["price"].to_numpy()
    y = test["target"].astype(int).to_numpy()
    depth = pd.cut(pd.Series(price), [-np.inf, -20, -5, -1, 0], right=False,
                   labels=["< -20", "-20..-5", "-5..-1", "-1..0"])
    rec, tol = {}, []
    for k in keys:
        pred = models[k]["pred"]
        hit = (y == 1)
        rec[k] = pd.Series(pred[hit], index=depth[hit].to_numpy()).groupby(level=0, observed=False).mean()
        f = pred == 1
        tol.append({"model": k, "flagged": int(f.sum()), "precision (<0)": (price[f] < 0).mean(),
                    "share <= 0": (price[f] <= 0).mean(), "share <= 1": (price[f] <= 1).mean(),
                    "share <= 5": (price[f] <= 5).mean()})
    order = ["< -20", "-20..-5", "-5..-1", "-1..0"]
    recall = pd.DataFrame(rec).reindex(order)
    recall.insert(0, "n_hours", pd.Series(depth[y == 1].to_numpy()).value_counts().reindex(recall.index))
    return recall, pd.DataFrame(tol).set_index("model")


def quarter_figure(qtab, test_y_by_quarter, keys):
    piv = qtab[qtab["model"].isin(keys)].pivot(index="subset", columns="model", values="pr_auc")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k in keys:
        if k in piv:
            ax.plot(piv.index, piv[k], marker="o", label=k)
    ax.set_xticks(range(len(piv.index)), [f"{q}\n(n_neg={test_y_by_quarter[q]})" for q in piv.index], fontsize=8)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Test period: PR-AUC by quarter (quarters with few negatives are noisy)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "pr_auc_by_quarter.png", dpi=200)


# ---------------------------------------------------------------------------
# Section 3: main
# ---------------------------------------------------------------------------
def main(n_boot, rerun):
    if MARK.exists() and not rerun:
        raise SystemExit(
            f"The final test run was already completed ({MARK}).\n"
            "Re-running after seeing the results breaks the protocol. Use --rerun only if you will say so in the paper."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(__doc__)

    data, _, _, nb_cols = load_data(with_extra=True)
    if not nb_cols:
        raise SystemExit("Neighbour features missing. Run `python src/neighbors.py` first.")
    data, n_filled = fill_neighbour_residual(data)
    test = get_split(data, "test", allow_test=True)
    test = test[test.index.tz_convert(LOCAL_TZ) < LAST_LOCAL_DAY]
    y = test["target"].astype(int).to_numpy()
    print(f"\nTEST PERIOD: {test.index.min()} .. {test.index.max()} | {len(test)} hours | {int(y.sum())} negative "
          f"({y.mean():.2%}) | neighbour values filled (all years): {n_filled}\n")

    models = build_models(data, test, nb_cols)

    # --- main table
    table = results_table([evaluate(y, m["pred"], y_score=m["score"], y_prob=m["prob"], name=n) for n, m in models.items()])
    print("\n=== TEST RESULTS (all test hours) ===")
    print(table.round(3).to_string())
    table.to_csv(OUT / "test_results.csv")

    # --- bootstrap and paired differences
    refs = ["B3 | rolling", "B5 | rolling", "B3 | static", "B5 | static", "LR | static", "XGB | static"]
    print(f"\nBootstrapping {n_boot} resamples of whole days ...")
    ci, diff = day_block_bootstrap(y, models, _day_ids(test.index), n_boot, ref_prefix=refs)
    ci.to_csv(OUT / "test_ci.csv", index=False)
    diff.to_csv(OUT / "test_paired_differences.csv", index=False)

    show = ci.assign(text=lambda d: d.apply(
        lambda r: "n/a" if np.isnan(r["estimate"]) else f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1))
    print("\n95% confidence intervals (estimate [low, high]):")
    print(show.pivot(index="model", columns="metric", values="text").loc[list(models)].to_string())

    pairs = [
        ("H1", "LR | rolling 12 months", "B3 | rolling 12 months"),
        ("H1", "XGB | rolling 12 months", "B3 | rolling 12 months"),
        ("H2", "LR | rolling 12 months", "B5 | rolling 12 months"),
        ("H2", "XGB | rolling 12 months", "B5 | rolling 12 months"),
        ("info", "B5 | rolling 12 months", "B3 | rolling 12 months"),
        ("H3", "LR | rolling 12 months", "LR | static (fit once)"),
        ("H3", "XGB | rolling 12 months", "XGB | static (fit once)"),
        ("H3", "B3 | rolling 12 months", "B3 | static (fit once)"),
        ("H3", "B5 | rolling 12 months", "B5 | static (fit once)"),
    ]
    rows = []
    for h, a, b in pairs:
        sel = diff[(diff["model"] == a) & (diff["vs"] == b)].copy()
        sel.insert(0, "hypothesis", h)
        rows.append(sel)
    key_diff = pd.concat(rows, ignore_index=True)
    print("\n=== PRE-DECLARED COMPARISONS (paired, positive = first model better) ===")
    print(key_diff.drop(columns=["share_draws_better"]).round(3).to_string(index=False))
    key_diff.to_csv(OUT / "test_key_comparisons.csv", index=False)

    # --- slices
    era0, era1 = test["flag_15min_era"].to_numpy() == 0, test["flag_15min_era"].to_numpy() == 1
    sl = pd.concat([
        subset_table(models, y, era0, "hourly era (before 1 Oct 2025)"),
        subset_table(models, y, era1, "15-minute era (from 1 Oct 2025)"),
        subset_table(models, y, test["nb_filled"].to_numpy() == 0, "complete neighbour data only"),
    ], ignore_index=True)
    cols = ["subset", "model", "n_hours", "n_negative", "precision", "recall", "f1", "pr_auc", "brier"]
    print("\n=== BY MARKET ERA AND DATA COMPLETENESS ===")
    print(sl[cols].round(3).to_string(index=False))
    sl[cols].to_csv(OUT / "test_by_era.csv", index=False)

    qtab = by_quarter(models, y, test)
    qtab.to_csv(OUT / "test_by_quarter.csv", index=False)
    loc = test.index.tz_convert(LOCAL_TZ)
    qkey = np.asarray(loc.year.astype(str) + "Q" + loc.quarter.astype(str))
    neg_by_q = {q: int(y[qkey == q].sum()) for q in sorted(set(qkey))}
    quarter_figure(qtab, neg_by_q, ["B3 | rolling 12 months", "B5 | rolling 12 months", "LR | static (fit once)",
                                    "LR | rolling 12 months", "XGB | static (fit once)", "XGB | rolling 12 months"])

    keys = ["B3 | rolling 12 months", "B5 | rolling 12 months", "LR | rolling 12 months", "XGB | rolling 12 months"]
    recall, tol = depth_tables(models, test, keys)
    print("\n=== RECALL BY DEPTH OF THE NEGATIVE PRICE ===")
    print(recall.round(3).to_string())
    print("\n=== PRECISION UNDER TOLERANCE (flagged hour counts if price <= X) ===")
    print(tol.round(3).to_string())
    recall.to_csv(OUT / "test_recall_by_depth.csv")
    tol.to_csv(OUT / "test_precision_tolerance.csv")

    # --- predictions and run record
    pred_df = pd.DataFrame({"y": y, "price": test["price"].to_numpy(),
                            "flag_15min_era": test["flag_15min_era"].to_numpy()}, index=test.index)
    for n, m in models.items():
        key = re.sub(r"[^0-9A-Za-z]+", "_", n).strip("_")
        pred_df[f"pred_{key}"] = m["pred"]
        pred_df[f"score_{key}"] = m["score"]
    pred_df.to_csv(OUT / "predictions_test.csv")

    record = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - t0) / 60, 1),
        "test_start": str(test.index.min()), "test_end": str(test.index.max()), "n_hours": int(len(test)),
        "n_negative": int(y.sum()), "decision_threshold": DECISION_THRESHOLD,
        "lr_params": best_params("LR"), "xgb_params": best_params("XGB"),
        "primary": PRIMARY, "secondary": SECONDARY, "bootstrap_resamples": n_boot,
    }
    json.dump(record, open(MARK, "w"), indent=2)
    print(f"\nDone in {record['minutes']} minutes. Everything is saved in {OUT}. The run-once guard is now set.")


if __name__ == "__main__":
    boots = int(sys.argv[sys.argv.index("--boots") + 1]) if "--boots" in sys.argv else 1000
    main(boots, "--rerun" in sys.argv)