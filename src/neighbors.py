"""
neighbors.py  --  day-ahead FORECASTS (load, solar, wind) of Germany's neighbours.

Why? German prices are set in a coupled European market. If France, the Netherlands, Denmark
etc. also have a sunny/windy surplus, German surplus power cannot be exported and prices go
negative; if neighbours are short of power, exports absorb the surplus. The German-only
features cannot see this.

Usage (from the repo root):
    python src/neighbors.py          # download (resumable) + coverage report

Leakage note: only day-ahead FORECASTS are fetched. Do NOT add scheduled cross-border flows
or neighbouring PRICES for the delivery day; those are auction results and would leak the target.
Same caveat as the German forecasts: publication times must be checked and reported.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_loader import EXPECTED_LAST, END, START  # noqa: E402

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "neighbors_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
NEIGHBORS_FILE = DATA_DIR / "neighbors.parquet"

# ENTSO-E area codes for Germany's electrical neighbours
COUNTRIES = ["FR", "NL", "BE", "DK_1", "DK_2", "AT", "CZ", "PL", "CH"]
MIN_COVERAGE = 0.98  # a series is used only if >= 98% of hours exist over the WHOLE period


TRANSIENT_ERRORS = {"HTTPError", "ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout", "ChunkedEncodingError"}


def _retry(fn, tries=3, wait=5):
    """Retry transient network/HTTP failures (3 tries). 'No data' errors are NOT retried."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ not in TRANSIENT_ERRORS or k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def _why(e):
    """Error name plus HTTP status code when there is one, e.g. 'HTTPError 503'."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    return type(e).__name__ + (f" {code}" if code else "")


# ---------------------------------------------------------------------------
# Section 1: download one country (cached per country so a crash can be resumed)
# ---------------------------------------------------------------------------
def _hourly(obj):
    """ENTSO-E series (hourly or 15-min) -> hourly mean in UTC."""
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 1:
        obj = obj.iloc[:, 0]
    return obj.tz_convert("UTC").resample("h").mean()


def fetch_country(client, cc, start=START, end=END):
    """Return an hourly DataFrame with columns load, solar, wind (missing ones are left out)."""
    out = {}
    try:
        out["load"] = _hourly(_retry(lambda: client.query_load_forecast(cc, start=start, end=end)))
    except Exception as e:  # noqa: BLE001  (API errors differ by country; report and go on)
        print(f"   {cc}: load forecast unavailable ({_why(e)})")
    try:
        ws = _retry(lambda: client.query_wind_and_solar_forecast(cc, start=start, end=end))
        ws = ws.tz_convert("UTC").resample("h").mean()
        if "Solar" in ws.columns:
            out["solar"] = ws["Solar"]
        wind_cols = [c for c in ws.columns if "Wind" in str(c)]
        if wind_cols:
            out["wind"] = ws[wind_cols].sum(axis=1, min_count=1)
    except Exception as e:  # noqa: BLE001
        print(f"   {cc}: wind/solar forecast unavailable ({_why(e)})")
    return pd.DataFrame(out)


def download_all(force=False):
    from entsoe import EntsoePandasClient

    load_dotenv()
    key = os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOES_API_KEY")
    if not key:
        raise ValueError("API key not found in .env")
    client = EntsoePandasClient(api_key=key)

    full_idx = pd.date_range(START, EXPECTED_LAST, freq="h", tz="UTC")
    frames = []
    for cc in COUNTRIES:
        cache = RAW_DIR / f"{cc}.parquet"
        if cache.exists() and not force:
            print(f"{cc}: using cached file")
            df = pd.read_parquet(cache)
            df.index = pd.to_datetime(df.index).tz_convert("UTC")
        else:
            print(f"{cc}: downloading ...")
            df = fetch_country(client, cc)
            df.to_parquet(cache)
        df = df.reindex(full_idx).ffill(limit=3)  # short gaps only, forward-fill (no future info)
        df.columns = [f"{cc}_{c}" for c in df.columns]
        frames.append(df)

    table = pd.concat(frames, axis=1)
    table.index.name = "timestamp_utc"
    table.to_parquet(NEIGHBORS_FILE)
    return table


def coverage_report(table):
    """Share of hours present per series. Anything below MIN_COVERAGE will be ignored."""
    cov = table.notna().mean().rename("coverage").to_frame()
    cov["used"] = cov["coverage"] >= MIN_COVERAGE
    return cov.round(3)


# ---------------------------------------------------------------------------
# Section 2: turn raw neighbour columns into model features
# ---------------------------------------------------------------------------
def build_neighbor_features(raw, used=None, verbose=False):
    """Build coupled-market features from a raw neighbour table. Returns (DataFrame, column_names).

    Per country: residual = load - solar - wind (using only the series in `used`; a country needs
    at least a load series). Then:
        nb_residual_sum : sum of neighbour residual loads
        nb_load_sum     : sum of neighbour loads
        nb_res_share    : nb_residual_sum / nb_load_sum
    `used` is a set of column names such as {"FR_load", "FR_solar"}. If None it is decided by the
    coverage check over the whole table (training). The live tool passes a fixed set instead.
    """
    if used is None:
        cov = coverage_report(raw)
        used = set(cov.index[cov["used"]])

    feats = pd.DataFrame(index=raw.index)
    loads, residuals, used_countries = [], [], []
    for cc in COUNTRIES:
        ok = {k: f"{cc}_{k}" in used for k in ("load", "solar", "wind")}
        if not ok["load"]:
            continue
        res = raw[f"{cc}_load"].copy()
        for k in ("solar", "wind"):
            if ok[k]:
                res = res - raw[f"{cc}_{k}"]
        feats[f"{cc}_residual"] = res
        loads.append(raw[f"{cc}_load"])
        residuals.append(res)
        used_countries.append(cc)

    if not used_countries:
        raise RuntimeError("No neighbour has usable load data; nothing to build.")
    feats["nb_residual_sum"] = pd.concat(residuals, axis=1).sum(axis=1, skipna=False)
    feats["nb_load_sum"] = pd.concat(loads, axis=1).sum(axis=1, skipna=False)
    feats["nb_res_share"] = feats["nb_residual_sum"] / feats["nb_load_sum"]
    if verbose:
        print("Countries used:", used_countries)
    return feats, list(feats.columns)


def neighbor_features(index, verbose=True):
    """Training-time version: read data/neighbors.parquet, apply the coverage rule, align to `index`."""
    if not NEIGHBORS_FILE.exists():
        raise FileNotFoundError("data/neighbors.parquet not found. Run: python src/neighbors.py")
    raw = pd.read_parquet(NEIGHBORS_FILE)
    raw.index = pd.to_datetime(raw.index).tz_convert("UTC")
    if verbose:
        print("Neighbour series coverage:")
        print(coverage_report(raw).to_string())
    feats, cols = build_neighbor_features(raw, verbose=verbose)
    return feats.reindex(index), cols


if __name__ == "__main__":
    table = download_all(force="--force" in sys.argv)
    print("\n" + coverage_report(table).to_string())
    print("\nSaved", NEIGHBORS_FILE)
    print("Next: python src/experiments.py features")