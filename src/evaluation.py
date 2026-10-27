"""
evaluation.py  --  one shared place for metrics, so baselines and ML models
are scored with EXACTLY the same code.

Why not accuracy?
    Only ~2-7% of hours are negative. A model that always says "not negative"
    scores 93-98% accuracy and is useless. We score the NEGATIVE class only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, y_score=None, y_prob=None, name="") -> dict:
    """Score one set of predictions on the negative class (label 1).

    y_pred  : hard 0/1 decisions (required)
    y_score : any continuous ranking score, higher = more likely negative.
              Used for PR-AUC. Optional.
    y_prob  : calibrated probabilities in [0, 1]. Used for the Brier score.
              Leave as None for hard rules, where Brier is not meaningful.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "model": name,
        "n_hours": len(y_true),
        "n_negative": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
        # zero_division=0: if a rule never flags anything, precision is 0, not an error
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if y_score is not None else np.nan,
        "brier": brier_score_loss(y_true, y_prob) if y_prob is not None else np.nan,
    }


def results_table(rows: list) -> pd.DataFrame:
    """List of evaluate() dicts -> tidy table."""
    return pd.DataFrame(rows).set_index("model")


def base_rate_row(y_true, name="always-negative (reference)") -> dict:
    """Reference line: PR-AUC of a random scorer equals the base rate.

    Any model's PR-AUC should be read against this number, not against 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    return {
        "model": name, "n_hours": len(y_true), "n_negative": int(y_true.sum()),
        "n_flagged": len(y_true), "precision": y_true.mean(), "recall": 1.0,
        "f1": f1_score(y_true, np.ones_like(y_true), zero_division=0),
        "pr_auc": y_true.mean(), "brier": np.nan,
    }