"""
baselines.py  --  the bar the ML models must clear (B1, B2, B3).

Usage (from the repo root):
    python src/baselines.py            # scores on VALIDATION (2024). Safe to run any time.
    python src/baselines.py --final    # scores on TEST. Run once, at the very end.

Every baseline is "free" of machine learning:
    B1  same hour yesterday      -> predict negative if yesterday's price at this hour was negative
    B2  same hour last week      -> same idea with a 7-day lag
    B3  residual-load threshold  -> predict negative if forecast residual load is below a cut-off
                                    chosen on the TRAINING data only
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

sys.path.append(str(Path(__file__).resolve().parent))
from evaluation import base_rate_row, evaluate, results_table  # noqa: E402
from features import make_dataset  # noqa: E402
from splits import add_flags, get_split  # noqa: E402

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Section 1: B3 threshold fitting (training data only)
# ---------------------------------------------------------------------------
def fit_residual_threshold(train_df):
    """Find the residual-load cut-off (MW) that maximises F1 on the training set.

    Lower residual load = more negative prices, so we use score = -residual_load
    (higher score = more likely negative) and sweep every possible cut-off with
    precision_recall_curve. Returns (threshold_MW, sweep_table_for_plotting).
    """
    y = train_df["target"].astype(int).to_numpy()
    score = -train_df["residual_load"].to_numpy()
    precision, recall, thr = precision_recall_curve(y, score)
    # precision/recall have one more entry than thr, so drop the last point
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    threshold_mw = -thr[best]  # convert back from score to residual load
    return threshold_mw, (-thr, f1)


# ---------------------------------------------------------------------------
# Section 2: predictions for each baseline
# ---------------------------------------------------------------------------
def predict_b1(df):
    """B1: negative if the same hour yesterday was negative."""
    pred = (df["price_lag_24"] < 0).astype(int)
    return pred, -df["price_lag_24"]  # score: lower past price -> higher score


def predict_b2(df):
    """B2: negative if the same hour last week was negative."""
    pred = (df["price_lag_168"] < 0).astype(int)
    return pred, -df["price_lag_168"]


def predict_b3(df, threshold_mw):
    """B3: negative if forecast residual load is at or below the threshold."""
    pred = (df["residual_load"] <= threshold_mw).astype(int)
    return pred, -df["residual_load"]


# ---------------------------------------------------------------------------
# Section 3: run everything on one evaluation split
# ---------------------------------------------------------------------------
def run(eval_name="val"):
    data = add_flags(make_dataset())
    train = get_split(data, "train")
    evl = get_split(data, eval_name, allow_test=(eval_name == "test"))

    # Fit B3 two ways: all training years, and excluding the 2022 crisis year
    thr_all, sweep = fit_residual_threshold(train)
    thr_no22, _ = fit_residual_threshold(train[train["flag_2022"] == 0])
    print(f"B3 threshold (all train years): {thr_all:,.0f} MW")
    print(f"B3 threshold (excluding 2022):  {thr_no22:,.0f} MW")

    y = evl["target"].astype(int)
    rows = []
    for name, (pred, score) in {
        "B1 same hour yesterday": predict_b1(evl),
        "B2 same hour last week": predict_b2(evl),
        f"B3 residual load <= {thr_all:,.0f} MW": predict_b3(evl, thr_all),
        f"B3b same, threshold fit without 2022 ({thr_no22:,.0f} MW)": predict_b3(evl, thr_no22),
    }.items():
        rows.append(evaluate(y, pred, y_score=score, name=name))
    rows.append(base_rate_row(y))

    table = results_table(rows)
    print(f"\nBaseline results on {eval_name.upper()} ({evl.index.min().date()} to {evl.index.max().date()}):")
    print(table.round(3).to_string())
    table.to_csv(RESULTS_DIR / f"baselines_{eval_name}.csv")

    # Figure: how F1 changes with the cut-off on the training data
    thr_axis, f1_axis = sweep
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thr_axis / 1000, f1_axis)
    ax.axvline(thr_all / 1000, color="red", ls="--", label=f"chosen: {thr_all/1000:.1f} GW")
    ax.set_xlim(np.percentile(thr_axis, 0.5) / 1000, np.percentile(thr_axis, 40) / 1000)
    ax.set_xlabel("Residual-load cut-off (GW)")
    ax.set_ylabel("F1 (negative class), training data")
    ax.set_title("B3: choosing the residual-load threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "b3_threshold_sweep.png", dpi=200)
    print(f"\nSaved {RESULTS_DIR / f'baselines_{eval_name}.csv'} and the threshold figure.")
    return table


if __name__ == "__main__":
    run("test" if "--final" in sys.argv else "val")