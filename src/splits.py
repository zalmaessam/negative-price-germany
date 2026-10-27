"""
splits.py  --  FROZEN time-based splits. Commit this file BEFORE building any model.

Why a separate file?
    The split boundaries are a protocol decision. Keeping them in one small,
    committed file means nobody (including future you) can quietly move them
    after seeing results.

All boundaries are LOCAL midnight in Europe/Berlin, converted to UTC. That way
a whole delivery day always lands in exactly one split (a UTC cut would slice
one auction day across two splits).
"""
import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"


def _local_midnight_utc(date_str: str) -> pd.Timestamp:
    """'2024-01-01' -> 00:00 Berlin time on that date, expressed in UTC."""
    return pd.Timestamp(date_str, tz=LOCAL_TZ).tz_convert("UTC")


# ---------------------------------------------------------------------------
# 1. THE FROZEN BOUNDARIES  (start inclusive, end exclusive)
# ---------------------------------------------------------------------------
TRAIN_START = _local_midnight_utc("2019-01-01")
VAL_START = _local_midnight_utc("2024-01-01")    # train = 2019-2023
TEST_START = _local_midnight_utc("2025-01-01")   # val = 2024
# test = 2025-01-01 .. end of data (30 Sept 2026)

# Special periods used for sensitivity analysis and RQ2
CRISIS_START = _local_midnight_utc("2022-01-01")
CRISIS_END = _local_midnight_utc("2023-01-01")
QUARTER_HOUR_START = _local_midnight_utc("2025-10-01")  # day-ahead moved to 15-min products


# ---------------------------------------------------------------------------
# 2. LABELLING EVERY ROW
# ---------------------------------------------------------------------------
def assign_split(index: pd.DatetimeIndex) -> pd.Series:
    """Return 'train' / 'val' / 'test' for every timestamp."""
    labels = np.where(
        index < VAL_START, "train", np.where(index < TEST_START, "val", "test")
    )
    return pd.Series(labels, index=index, name="split")


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add split label plus two indicator columns used for robustness checks.

    flag_2022      : 1 during the energy-crisis year (report results with and without)
    flag_15min_era : 1 from 1 Oct 2025, when the market switched to 15-min products
    These are for ANALYSIS only. Do not feed them to the models as features.
    """
    out = df.copy()
    out["split"] = assign_split(out.index)
    out["flag_2022"] = ((out.index >= CRISIS_START) & (out.index < CRISIS_END)).astype(int)
    out["flag_15min_era"] = (out.index >= QUARTER_HOUR_START).astype(int)
    return out


# ---------------------------------------------------------------------------
# 3. ACCESS CONTROL: the test set is "touched once"
# ---------------------------------------------------------------------------
def get_split(df: pd.DataFrame, name: str, allow_test: bool = False) -> pd.DataFrame:
    """Return one split. The test set is locked unless you pass allow_test=True.

    The lock is a speed bump, not security: it forces a deliberate decision
    every time you ask for the test data. Use it exactly once, at the end.
    """
    if name not in ("train", "val", "test"):
        raise ValueError("name must be 'train', 'val' or 'test'")
    if name == "test" and not allow_test:
        raise PermissionError(
            "Test set is frozen. Pass allow_test=True only for the final evaluation."
        )
    if "split" not in df.columns:
        df = add_flags(df)
    return df[df["split"] == name]


# ---------------------------------------------------------------------------
# 4. CROSS-VALIDATION INSIDE TRAIN (expanding window, one calendar year per fold)
# ---------------------------------------------------------------------------
def walk_forward_folds(train_df: pd.DataFrame, first_val_year: int = 2020):
    """Yield (fit_idx, val_idx, val_year) with an expanding training window.

    Fold 1: fit 2019            -> validate 2020
    Fold 2: fit 2019-2020       -> validate 2021
    Fold 3: fit 2019-2021       -> validate 2022 (crisis year, flagged)
    Fold 4: fit 2019-2022       -> validate 2023

    Always trains on the past and validates on the future. No random shuffling.
    Years are LOCAL calendar years so whole delivery days stay together.
    """
    local_year = train_df.index.tz_convert(LOCAL_TZ).year
    last_year = local_year.max()
    for year in range(first_val_year, last_year + 1):
        fit_idx = train_df.index[local_year < year]
        val_idx = train_df.index[local_year == year]
        if len(fit_idx) and len(val_idx):
            yield fit_idx, val_idx, year


# ---------------------------------------------------------------------------
# 5. SUMMARY (paste into the paper's data section)
# ---------------------------------------------------------------------------
def summary(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Rows, date range and negative-hour rate per split."""
    if "split" not in df.columns:
        df = add_flags(df)
    g = df.groupby("split")
    out = pd.DataFrame(
        {
            "rows": g.size(),
            "start": g.apply(lambda x: x.index.min()),
            "end": g.apply(lambda x: x.index.max()),
            "neg_rate_%": g[target_col].mean() * 100,
        }
    )
    return out.loc[["train", "val", "test"]]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent))
    from features import make_dataset

    data = add_flags(make_dataset())
    print(summary(data))
    print("\nWalk-forward folds inside train:")
    for fit_idx, val_idx, year in walk_forward_folds(get_split(data, "train")):
        print(f"  fit {fit_idx.min().date()}..{fit_idx.max().date()} -> validate {year} ({len(val_idx)} rows)")