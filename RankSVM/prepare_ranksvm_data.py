#!/usr/bin/env python3
"""
prepare_ranksvm_data.py -- make RankSVM-ready copies of the JEE and reco_thurstone datasets.

What RankSVM/Scripts/run_svmrank.py needs from a CSV
  * the file sits in RankSVM/Datasets/ and every --features column is NUMERIC (text columns make every row "malformed");
  * rows are listed best-first by the ranking column, because --n keeps the FIRST n rows of the file
    (the script re-sorts only when the ranking column is literally called 'mark');
  * a ranking column whose direction you pass with --higher_is_better (omit it when rank 1 is the best).

The copies are built to match exactly what main.py feeds to LP-PP, so both methods see the same feature matrix:
  JEE   : rows kept in the FILE ORDER (the given ranking, row 0 = rank 1; main.py, hp_inv.py and ls_inv.py no longer re-sort
          by mark) and numbered in a `rank` column (1 = best: use --ranking rank and OMIT --higher_is_better; do NOT use
          --ranking mark, run_svmrank.py would re-sort by mark); category GEN/OBC-NCL/SC/ST -> 0/1/2/3, GENDER M/F -> 0/1;
          math, phys, chem min-max scaled over the whole file (main.py's default --jee_scale 1); mark is not written.
  reco  : sorted by rank ascending (stable); only the 7 visible features + rank are kept (no hidden_score, visible_score,
          boost_signal, diversity_penalty_signal, item_id, is_sponsored or ranking_model); features are min-max scaled
          over the whole file by default (--reco_scale 1). IMPORTANT: main.py reads reco UNSCALED, but SVMrank does not
          converge on the raw reco scales (popularity reaches ~300, price ~200, the rest 0-1 or 0-7): in tests a run that
          finishes in 2 s on scaled features was still at iteration 16 after 250 s on raw ones. With scaling, PP is
          measured in scaled units and is not numerically comparable with LP-PP's raw-unit PP. --reco_scale 0 writes raw.

    python prepare_ranksvm_data.py --jee data/jee.csv --reco data/reco_thurstone.csv --out RankSVM/Datasets/
    python prepare_ranksvm_data.py --jee data/jee.csv --jee_scale 0          # keep raw marks instead
    python prepare_ranksvm_data.py --reco data/reco_thurstone.csv --reco_top 100000   # keep only the best 100,000 items
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

CATEGORY = {"GEN": 0, "OBC-NCL": 1, "SC": 2, "ST": 3}
GENDER = {"M": 0, "F": 1}
JEE_FEATURES = ["category", "GENDER", "math", "phys", "chem"]
RECO_FEATURES = ["relevance_score", "popularity", "freshness", "predicted_ctr", "quality_score", "price", "diversity_tag"]


def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit(f"{name}: missing column(s) {miss}. Columns found: {list(df.columns)}")


def prep_jee(src, out_dir, scale):
    df = pd.read_csv(src, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    need(df, JEE_FEATURES + ["mark"], src)
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df = df.dropna(subset=["mark"]).reset_index(drop=True)          # the file order IS the ranking: no sort by mark
    cat = df["category"].astype(str).str.strip().str.upper().map(CATEGORY)
    gen = df["GENDER"].astype(str).str.strip().str.upper().map(GENDER)
    for name, col, raw in (("category", cat, df["category"]), ("GENDER", gen, df["GENDER"])):
        if col.isna().any():
            bad = sorted(raw[col.isna()].astype(str).str.strip().str.upper().unique())[:8]
            sys.exit(f"{src}: {int(col.isna().sum())} rows have a {name} value that is not encoded: {bad}")
    res = pd.DataFrame({"category": cat.astype(int), "GENDER": gen.astype(int)})
    for c in ("math", "phys", "chem"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            sys.exit(f"{src}: column {c} has {int(v.isna().sum())} non-numeric values")
        if scale:                                    # same arithmetic as sklearn's MinMaxScaler, fitted on the whole file
            lo, hi = float(v.min()), float(v.max())
            s = 1.0 / (hi - lo)
            v = v.astype(float) * s + (-lo * s)
        res[c] = v
    res["rank"] = range(1, len(res) + 1)                              # 1 = best = first row of the file
    path = os.path.join(out_dir, "jee_fileorder_ranksvm.csv")
    res.to_csv(path, index=False)
    print(f"JEE  -> {path}   {len(res):,} rows in file order (rank column 1..n), features scaled: {bool(scale)}")
    return path, len(res)


def prep_reco(src, out_dir, top, scale):
    df = pd.read_csv(src, usecols=["rank"] + RECO_FEATURES, low_memory=False)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"]).sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)
    if df[RECO_FEATURES].isna().any().any():
        sys.exit(f"{src}: NaN values in the feature columns")
    if scale:                                        # min-max over the whole file, same arithmetic as sklearn's MinMaxScaler
        for c in RECO_FEATURES:
            lo, hi = float(df[c].min()), float(df[c].max())
            if hi > lo:
                sc = 1.0 / (hi - lo)
                df[c] = df[c].astype(float) * sc + (-lo * sc)
            else:
                df[c] = 0.0
    if top:
        df = df.head(top)
    res = df[RECO_FEATURES + ["rank"]]
    path = os.path.join(out_dir, "reco_thurstone_ranksvm.csv")
    res.to_csv(path, index=False)
    assert res["rank"].is_monotonic_increasing
    print(f"reco -> {path}   {len(res):,} rows, sorted by rank (ascending), 7 features + rank, features scaled: {bool(scale)}")
    if scale:
        print("       note: RankSVM only converges on scaled reco features; PP is then in scaled units (main.py reads reco raw).")
    return path, len(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jee", help="jee.csv (columns REGST_NO, NAME, category, GENDER, math, phys, chem, mark)")
    ap.add_argument("--reco", help="reco_thurstone.csv")
    ap.add_argument("--out", default="RankSVM/Datasets/", help="output folder (RankSVM's Datasets folder)")
    ap.add_argument("--jee_scale", type=int, default=1, help="1 = min-max scale math/phys/chem like main.py (default); 0 = raw")
    ap.add_argument("--reco_top", type=int, default=None, help="keep only the best N items of the reco file (default: all)")
    ap.add_argument("--reco_scale", type=int, default=1, help="1 = min-max scale the 7 reco features (default; SVMrank needs it); 0 = raw")
    a = ap.parse_args()
    if not (a.jee or a.reco):
        sys.exit("give --jee and/or --reco")
    os.makedirs(a.out, exist_ok=True)
    print()
    if a.jee:
        prep_jee(a.jee, a.out, a.jee_scale)
    if a.reco:
        prep_reco(a.reco, a.out, a.reco_top, a.reco_scale)
    print("\nCommands (run from the RankSVM folder; add --n and --k as needed):")
    if a.jee:
        print("  python Scripts/run_svmrank.py --dataset jee_fileorder_ranksvm.csv --features category GENDER math phys chem "
              "--ranking rank --mode topk --k 1000 --n 10000 --C 20")
    if a.reco:
        print("  python Scripts/run_svmrank.py --dataset reco_thurstone_ranksvm.csv --features relevance_score popularity "
              "freshness predicted_ctr quality_score price diversity_tag --ranking rank --mode topk --k 100 --n 1000")


if __name__ == "__main__":
    main()
