#!/usr/bin/env python3
"""
make_derived_datasets.py -- build the datasets for the "derived attributes" experiments (E1, E2, E3) WITHOUT touching main.py.

Each output CSV has the columns  institution, rank, score, <features...>.
main.py's `csrankings` loader reads exactly this layout: the reference order is `rank` ascending, `institution`, `rank` and
`score` are metadata, every other column is a feature (not normalized).  So every file runs with

    python main.py --dataset csrankings --csrankings_csv <file> --items N --tk K --lppp 1 --dppp 0 \
                   --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 0

Datasets
  E1  CSRankings  cs_raw     the 27 area counts (identical to what main.py's csrankings loader uses)
                  cs_log     log(1 + count) of every area                      (27)
                  cs_rawlog  raw and log together                              (54)
      CSRankings ranks by the geometric mean of (count + 1) over the areas, i.e. exp(mean(log(1 + count))): the log
      attributes make the ranking exactly linear.  The script checks this identity on your file.
  E2  Synthetic   syn_raw    the 7 visible features of the recommendation data (identical to main.py's reco loader, scale=False)
                  syn_oh     + one-hot encoding of diversity_tag                (7 + 8 = 15)
                  syn_ohpop  + one-hot(tag) x min-max popularity                (7 + 8 + 8 = 23)
      Only the best --syn_top items by rank are written (the file has 1,000,000); popularity is min-max scaled over the
      whole file, as the generator does.  Works for any file in the same format (reco_1 / reco_thurstone / reco_mallows).
  E3  NBA         nba_raw    the 8 statistics, min-max scaled over all players (identical to main.py's get_nba_dataset)
                  nba_sq     + squares                                          (8 + 8 = 16)
                  nba_poly2  + squares and all pairwise products                (8 + 8 + 28 = 44)
      rank = position in the file (sorted by MP*PER), as in the NBA loader.

    python make_derived_datasets.py --csrankings data/csrankings_full.csv --reco data/reco_1.csv --nba data/per.csv --out data/derived/
    python make_derived_datasets.py --csrankings data/csrankings_full.csv --out data/derived/        # only E1
    python make_derived_datasets.py --reco data/reco_thurstone.csv --syn_prefix thurstone --syn_top 10000 --out data/derived/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

RECO_FEATURES = ["relevance_score", "popularity", "freshness", "predicted_ctr", "quality_score", "price", "diversity_tag"]
CMD = ("python main.py --dataset csrankings --csrankings_csv {path} --items {n} --tk {k} --lppp 1 --dppp 0 "
       "--outside_vs_mintop 1 --incremental_topk 1 --earlybreak 0")
written = []                                         # (name, path, rows, d)


def need(cols, have, where):
    miss = [c for c in cols if c not in have]
    if miss:
        sys.exit(f"{where}: missing column(s) {miss}. Columns found: {list(have)[:20]}")


def save(out_dir, name, institution, rank, score, F, cols):
    """write institution, rank, score + features and check the file by reading it back"""
    F = np.asarray(F, float)
    if not np.isfinite(F).all():
        sys.exit(f"{name}: non-finite feature values")
    meta = pd.DataFrame({"institution": np.asarray(institution), "rank": np.asarray(rank), "score": np.asarray(score)})
    out = pd.concat([meta, pd.DataFrame(F, columns=cols)], axis=1)
    path = os.path.join(out_dir, name + ".csv")
    out.to_csv(path, index=False)
    back = pd.read_csv(path, nrows=5)
    assert list(back.columns) == ["institution", "rank", "score"] + list(cols), name
    written.append((name, path, len(out), len(cols)))


# ----------------------------------------------------------------------------- E1
def make_cs(src, out_dir):
    c = pd.read_csv(src)
    c.columns = [str(x).strip() for x in c.columns]
    need(["institution", "rank"], c.columns, src)
    areas = [x for x in c.columns if x not in ("institution", "rank", "score")]
    X = c[areas].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if np.isnan(X).any() or (X < 0).any():
        sys.exit(f"{src}: area columns must be numeric and non-negative")
    L = np.log1p(X)
    if "score" in c.columns:
        score = pd.to_numeric(c["score"], errors="coerce").to_numpy(float)
        ok = np.allclose(np.exp(L.mean(axis=1)), score, rtol=0, atol=1e-9)
        print(f"CSRankings: score == exp(mean(log(1 + area counts))) on all {len(c)} institutions: {ok}")
        if not ok:
            print("  WARNING: this file's score is not the geometric mean of (count + 1); the log attributes will not make the "
                  "ranking exactly linear.")
    else:
        score = np.full(len(c), np.nan)
    inst, rank = c["institution"], c["rank"]
    logc = [f"log_{a}" for a in areas]
    save(out_dir, "cs_raw", inst, rank, score, X, areas)
    save(out_dir, "cs_log", inst, rank, score, L, logc)
    save(out_dir, "cs_rawlog", inst, rank, score, np.hstack([X, L]), areas + logc)


# ----------------------------------------------------------------------------- E2
def make_syn(src, out_dir, top, prefix):
    hdr = list(pd.read_csv(src, nrows=0).columns)
    hdr = [str(h).strip() for h in hdr]
    need(["rank"] + RECO_FEATURES, hdr, src)
    meta_col = "hidden_score" if "hidden_score" in hdr else "rank"
    d = pd.read_csv(src, usecols=lambda c: str(c).strip() in set(["item_id", "rank", meta_col] + RECO_FEATURES))
    d.columns = [str(c).strip() for c in d.columns]
    if "item_id" not in d.columns:
        d["item_id"] = np.arange(len(d))
    d["rank"] = pd.to_numeric(d["rank"], errors="coerce")
    d = d.dropna(subset=["rank"])
    pmin, pmax = float(d["popularity"].min()), float(d["popularity"].max())          # whole-file min-max, as in the generator
    tags = sorted(int(t) for t in d["diversity_tag"].unique())
    d = d.sort_values("rank", ascending=True, kind="mergesort").reset_index(drop=True)
    if top > len(d):
        sys.exit(f"--syn_top {top} is larger than the file ({len(d):,} rows)")
    d = d.head(top)
    pop = ((d["popularity"] - pmin) / (pmax - pmin)).to_numpy(float)
    tag = d["diversity_tag"].astype(int).to_numpy()
    oh = np.column_stack([(tag == t).astype(float) for t in tags])
    ohpop = oh * pop[:, None]
    base = d[RECO_FEATURES].to_numpy(float)
    ocols = [f"tag_{t}" for t in tags]
    pcols = [f"tag_{t}_x_pop" for t in tags]
    args = (d["item_id"], d["rank"], d[meta_col])
    save(out_dir, f"{prefix}_raw", *args, base, RECO_FEATURES)
    save(out_dir, f"{prefix}_oh", *args, np.hstack([base, oh]), RECO_FEATURES + ocols)
    save(out_dir, f"{prefix}_ohpop", *args, np.hstack([base, oh, ohpop]), RECO_FEATURES + ocols + pcols)
    print(f"Synthetic ({os.path.basename(src)}): top {top:,} items by rank, diversity_tag values {tags}")


# ----------------------------------------------------------------------------- E3
def make_nba(src, out_dir):
    df = pd.read_csv(src, skiprows=1, header=0).dropna()            # skips the title line, drops the trailing 'End NBA' row
    feats = list(df.columns[1:9])
    M = df[feats].to_numpy(float)
    lo, hi = M.min(axis=0), M.max(axis=0)                            # min-max over all players with sklearn's MinMaxScaler arithmetic,
    scale = 1.0 / (hi - lo)                                          # so nba_raw equals get_nba_dataset() bit for bit
    M = M * scale + (-lo * scale)
    sq = M ** 2
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    pr = np.column_stack([M[:, i] * M[:, j] for i, j in pairs])
    util = df["MP*PER"].to_numpy(float) if "MP*PER" in df.columns else np.full(len(df), np.nan)
    if "MP*PER" in df.columns and not (np.diff(util) <= 0).all():
        print("  WARNING: the NBA file is not sorted by MP*PER (descending); the NBA loader uses the file order as the ranking.")
    inst, rank = df.iloc[:, 0].to_numpy(), np.arange(1, len(df) + 1)
    sqc = [f"sq_{f}" for f in feats]
    prc = [f"{feats[i]}*{feats[j]}" for i, j in pairs]
    save(out_dir, "nba_raw", inst, rank, util, M, feats)
    save(out_dir, "nba_sq", inst, rank, util, np.hstack([M, sq]), feats + sqc)
    save(out_dir, "nba_poly2", inst, rank, util, np.hstack([M, sq, pr]), feats + sqc + prc)
    print(f"NBA: {len(df):,} players, features {feats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csrankings", help="csrankings_full.csv (columns institution, rank, score, 27 areas)")
    ap.add_argument("--reco", help="synthetic recommendation CSV, e.g. reco_1.csv (hidden-score model)")
    ap.add_argument("--nba", help="per.csv (title line, header, players, 'End NBA')")
    ap.add_argument("--out", default="data/derived/", help="output folder (default data/derived/)")
    ap.add_argument("--syn_top", type=int, default=10000, help="number of best-ranked synthetic items to write (default 10000)")
    ap.add_argument("--syn_prefix", default="syn", help="file prefix for the synthetic sets (default syn)")
    a = ap.parse_args()
    if not (a.csrankings or a.reco or a.nba):
        sys.exit("give at least one of --csrankings, --reco, --nba")
    for p in (a.csrankings, a.reco, a.nba):
        if p and not os.path.isfile(p):
            sys.exit(f"file not found: {p}")
    os.makedirs(a.out, exist_ok=True)
    print()
    if a.csrankings:
        make_cs(a.csrankings, a.out)
    if a.reco:
        make_syn(a.reco, a.out, a.syn_top, a.syn_prefix)
    if a.nba:
        make_nba(a.nba, a.out)
    print(f"\n{'file':32s} {'rows':>8} {'features':>9}")
    for name, path, n, d in written:
        print(f"{path:32s} {n:>8,} {d:>9}")
    print("\nrun each file with (set N <= rows and K < N):")
    shown = set()
    for name, path, n, d in written:
        grp = name.split("_")[0]
        if grp not in shown:
            shown.add(grp)
            print("  " + CMD.format(path=path, n=min(n, 10000 if grp != "cs" else n), k={"cs": 50, "syn": 50, "nba": 50}.get(grp, 50)))


if __name__ == "__main__":
    main()
