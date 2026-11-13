import time
import gurobipy as gp
import numpy as np
import logging
from itertools import combinations
import math
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random
import itertools
import argparse


file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)



def _normalize_tiers(topk_ranking):
    """set -> single tie tier; list -> linear order; list of lists -> tiers as given."""
    def _is_intlike(x):
        try:
            import numpy as _np
            return isinstance(x, (int,)) or isinstance(x, _np.integer)
        except Exception:
            return isinstance(x, int)
    if isinstance(topk_ranking, set):
        return [list(int(i) for i in topk_ranking)]
    elif hasattr(topk_ranking, "__iter__") and all(_is_intlike(x) for x in topk_ranking):
        return [[int(i)] for i in topk_ranking]
    else:
        return [[int(i) for i in tier] for tier in topk_ranking]

def lppp_diagnostics_counts(points, topk_ranking, alpha_unit, *, strict=False, epsilon=1e-4, tol=1e-12):
    """
    Returns:
      {
        "pp": float,                           # raw sum of positive parts (score gaps)
        "topk_inversions_count": int,          # earlier (better) scored worse than later (worse)
        "outside_pair_violations": int,        # count of (outsider, topk) with s[o]-s[t]+margin > tol
        "outsiders_with_any_violation": int,   # number of distinct outsiders with any such violation
        "max_abs_rank_diff": int,              # max |pos_score - pos_given| over top-k items
        "sum_abs_rank_diff": int               # sum |pos_score - pos_given| over top-k items
      }
    """
    n, dims = len(points), len(points[0])
    assert len(alpha_unit) == dims, "alpha dimension mismatch"

    scores = [sum(alpha_unit[d]*points[i][d] for d in range(dims)) for i in range(n)]
    if args.verbose:
        print("scores", scores)

    tiers = _normalize_tiers(topk_ranking)
    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = [i for i in range(n) if i not in topk_set]

    margin = 0.0 if strict else epsilon

    # ----- existing violation metrics  -----
    count_pp = 0.0
    inversions = 0
    seen = []
    for tier in tiers:
        for t in tier:               # later/worse
            for c in seen:           # earlier/better
                gap = scores[t] - scores[c]
                if gap > 0:
                    if gap + margin > tol:
                        inversions += 1
                        count_pp += gap
        seen.extend(tier)

    pair_viol = 0
    outsider_any = set()
    for o in outsiders:
        so = scores[o]
        violated = False
        for t in ordered_topk:
            gap = so - scores[t]
            if gap > 0:
                if gap + margin > tol:
                    pair_viol += 1
                    count_pp += gap
                    violated = True
        if violated:
            outsider_any.add(o)

    # ----- ranking discrepancy between score ranking and given top-k order -----
    score_ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
    pos_score = {i: r for r, i in enumerate(score_ranking)}
    pos_given = {i: r for r, i in enumerate(ordered_topk)}  
    if args.verbose:
        print("pos_score", pos_score)
        print("pos_given", pos_given)

    abs_diffs = [abs(pos_score[i] - pos_given[i]) for i in ordered_topk]
    max_abs_rank_diff = max(abs_diffs) if abs_diffs else 0
    sum_abs_rank_diff = int(sum(abs_diffs)) if abs_diffs else 0

    return {
        "pp": count_pp,
        "topk_inversions_count": inversions,
        "outside_pair_violations": pair_viol,
        "full_inv": inversions + pair_viol,
        "outsiders_with_any_violation": len(outsider_any),
        "max_abs_rank_diff": max_abs_rank_diff,
        "sum_abs_rank_diff": sum_abs_rank_diff,
    }


def get_nba_dataset():
    df = pd.read_csv('./data/per.csv', skiprows=1, header=0).dropna()
    df = df[df.columns[1:9]]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled.values.tolist()


def _scores_from_weights(points_np: np.ndarray, w: np.ndarray) -> np.ndarray:
    return points_np @ w

def _ranking_from_scores(scores: np.ndarray) -> list:
    return np.argsort(-scores).tolist()

def _kendall_topk_error_from_ranking(ranking: list, ground_truth_topk: list) -> int:
    """Pairs inside top-k in wrong order + outsiders above any top-k."""
    ground_truth_set = set(ground_truth_topk)
    pos = {item: p for p, item in enumerate(ranking)}
    inv = 0
    # type1: order inside top-k
    for i in range(len(ground_truth_topk)):
        for j in range(i+1, len(ground_truth_topk)):
            a = ground_truth_topk[i]
            b = ground_truth_topk[j]
            if pos[a] > pos[b]:
                inv += 1
    # type2: outsider outranking top-k
    for p, item in enumerate(ranking):
        if item not in ground_truth_set:
            for gt in ground_truth_topk:
                if pos[gt] > p:
                    inv += 1
    return inv

def _is_ranking_realisable(points_np: np.ndarray, ranking: list, epsilon: float=1e-4):
    
    n, d = points_np.shape
    m = gp.Model()
    m.Params.OutputFlag = 0
    w = m.addMVar(shape=d, lb=epsilon, ub=1000.0)
    for i in range(len(ranking)-1):
        a = ranking[i]
        b = ranking[i+1]
        diff = points_np[a] - points_np[b]
        m.addConstr(w @ diff >= epsilon)
    m.addConstr(w.sum() == 1.0)
    m.setObjective(w.sum(), gp.GRB.MINIMIZE)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        return True, np.array([w[i].X for i in range(d)])
    return False, np.zeros(d)

def greedy_refine_weights(points_np: np.ndarray,
                          initial_w: np.ndarray,
                          ground_truth_topk: list,
                          verbose: bool=False):
    """Greedy adjacent swaps: accept if error improves AND realizable; update weights to the LP solution."""
    w = initial_w.copy()
    scores = _scores_from_weights(points_np, w)
    ranking = _ranking_from_scores(scores)
    best_err = _kendall_topk_error_from_ranking(ranking, ground_truth_topk)
    if verbose:
        print(f"Initial error: {best_err}")

    improved = True
    it = 0
    lp_calls = 0
    while improved:
        improved = False
        it += 1
        for i in range(len(ranking)-1):
            new_r = ranking.copy()
            new_r[i], new_r[i+1] = new_r[i+1], new_r[i]
            new_err = _kendall_topk_error_from_ranking(new_r, ground_truth_topk)
            if new_err >= best_err:
                continue
            # Only now check realizability
            feasible, w_cand = _is_ranking_realisable(points_np, new_r)
            lp_calls += 1
            if not feasible:
                continue
            # accept
            ranking = new_r
            w = w_cand
            best_err = new_err
            if verbose:
                print(f"Iteration {it}: improvement -> error {best_err}")
            improved = True
            break
    if verbose:
        print(f"Reached local minimum after {it} iterations; LP calls: {lp_calls}")
    return w, best_err

def _random_positive_unit_weights(d: int, rng: np.random.RandomState) -> np.ndarray:
    w = np.abs(rng.randn(d))
    s = w.sum()
    if s == 0:
        w = np.ones(d) / d
    else:
        w = w / s
    return w

def random_sampling_best(points_np: np.ndarray,
                         ground_truth_topk: list,
                         samples: int,
                         seed: int=None,
                         verbose: bool=False):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n, d = points_np.shape
    best_w = None
    best_err = float('inf')
    for _ in range(samples):
        w = _random_positive_unit_weights(d, rng)
        err = _kendall_topk_error_from_ranking(_ranking_from_scores(points_np @ w), ground_truth_topk)
        if err < best_err:
            best_err = err
            best_w = w
    if verbose:
        print(f"Random sampling best error: {best_err}")
    return best_w, best_err


def get_jee_dataset(csv_path: str, drop_cols=None, scale=False):
    """
    Reads a JEE CSV and returns a numeric feature matrix (list of lists),
    keeping marks and adding encoded category/GENDER/is_pwd.

    Encodings:
      - GENDER: M -> 0, F -> 1
      - category: GEN -> 0, OBC-NCL -> 1, SC -> 2, ST -> 3
      - is_pwd: True/Yes/1 -> 1, False/No/0 -> 0
    """
    if drop_cols is None:
        drop_cols = ["RANK", "BLOCK_ID", "AIR", "REGST_NO", "mark"]

    # Read with stable inference
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["__orig_idx__"] = np.arange(len(df))
    if "mark" not in df.columns:
        raise ValueError("JEE dataset must contain the target column 'mark'.")
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df = df.dropna(subset=["mark"]).reset_index(drop=True)
    df["__sorted_rank__"] = np.arange(len(df))

    # --- Clean & encode categorical ---
    if "category" in df.columns:
        cat_map = {"GEN": 0, "OBC-NCL": 1, "SC": 2, "ST": 3}
        df["category"] = (
            df["category"].astype(str).str.strip().str.upper().map(cat_map)
        )

    if "GENDER" in df.columns:
        gender_map = {"M": 0, "F": 1}
        df["GENDER"] = (
            df["GENDER"].astype(str).str.strip().str.upper().map(gender_map)
        )

    if "is_pwd" in df.columns:
        def _to_bin(x):
            if pd.isna(x): return np.nan
            if isinstance(x, str):
                s = x.strip().lower()
                if s in {"1","true","t","yes","y"}: return 1.0
                if s in {"0","false","f","no","n"}: return 0.0
            try:
                return 1.0 if bool(int(x)) else 0.0
            except Exception:
                return 1.0 if bool(x) else 0.0
        df["is_pwd"] = df["is_pwd"].apply(_to_bin)

   
    for col in ["math", "phys", "chem", "mark"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

   
    to_drop = [c for c in drop_cols if c in df.columns and c != "__orig_idx__"]
    df = df.drop(columns=to_drop, errors="ignore")

    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        raise ValueError("No numeric columns found after encoding and dropping IDs.")

    
    encoded_cols = [c for c in ["category", "GENDER"] if c in num_df.columns]
    cont_cols = [c for c in ["math", "phys", "chem"] if c in num_df.columns]
    if scale and cont_cols:
        scaler = MinMaxScaler()
        num_df[cont_cols] = scaler.fit_transform(num_df[cont_cols].values)

   
    num_df = num_df.sort_values("__sorted_rank__")

    
    ordered = [c for c in ["category", "GENDER",  "math", "phys", "chem"] if c in num_df.columns]
    out = num_df[ordered].to_numpy().tolist()

   
    return out


def get_reco_dataset(csv_path: str, scale=False):
    """
    Reads the synthetic recommendation CSV and returns only visible numeric features.

    Excludes:
      - rank
      - hidden_score
      - visible_score
      - item_id

    Also excludes hidden/debug columns if present:
      - boost_signal
      - diversity_penalty_signal
      - is_sponsored
      - is_new
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "rank" not in df.columns:
        raise ValueError("Recommendation dataset must contain a 'rank' column.")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"]).sort_values(
        "rank", ascending=True, kind="mergesort"
    ).reset_index(drop=True)

    drop_cols = [
        "rank",
        "hidden_score",
        "visible_score",
        "item_id",
        "boost_signal",
        "diversity_penalty_signal",
        "is_sponsored",
        "is_new",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # keep only numeric columns
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        raise ValueError("No numeric columns found in recommendation dataset after dropping score/rank columns.")

    if scale:
        scaler = MinMaxScaler()
        num_df[:] = scaler.fit_transform(num_df.values)
    return num_df.to_numpy().tolist()


def get_online_news_dataset(csv_path: str, scale=True, feature_set="all"):
    """Load UCI Online News Popularity as a ranking dataset.

    Rows are sorted by shares descending, so row 0 is the most-shared article.
    The target (shares), URL, and timedelta are never used as input features.

    feature_set="all": use all 58 predictive features.
    feature_set="all": exclude keyword-share and self-reference-share
        aggregates, which encode historical popularity information.
    """
    df = pd.read_csv(csv_path, low_memory=False, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]

    required = {"shares", "url", "timedelta"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Online News CSV is missing expected columns: {sorted(missing)}. "
            "Check that --news_csv points to OnlineNewsPopularity.csv."
        )

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df.dropna(subset=["shares"]).copy()

    # Keep article-level predictive features only; don't use target/ID/time.
    excluded = {"url", "timedelta", "shares"}
    feature_cols = [c for c in df.columns if c not in excluded]

    if feature_set == "article_only":
        feature_cols = [
            c for c in feature_cols
            if not c.startswith("kw_") and not c.startswith("self_reference_")
        ]
    elif feature_set != "all":
        raise ValueError("feature_set must be 'all' or 'article_only'")

    # Ensure numeric feature matrix and consistent column order.
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & np.isfinite(X.to_numpy()).all(axis=1)
    df = df.loc[valid].copy()
    X = X.loc[valid].copy()

    # Stable descending order gives a reproducible reference ranking by shares.
    order = df["shares"].sort_values(ascending=False, kind="mergesort").index
    X = X.loc[order]
    shares_sorted = df.loc[order, "shares"]

    if scale:
        scaler = MinMaxScaler()
        X.loc[:, :] = scaler.fit_transform(X.to_numpy(dtype=float))

    print(f"Online News Popularity rows: {len(X)}")
    print(f"Online News Popularity features: {X.shape[1]} ({feature_set})")
    print(f"Reference ranking: shares descending; top shares={shares_sorted.iloc[0]:g}")
    return X.to_numpy(dtype=float).tolist()


def get_csrankings_dataset(csv_path: str, scale=False):
    """Load CSRankings institution-level area scores as a ranking dataset.

    The reference order is rank ascending when a rank column is present;
    otherwise score descending when a score column is present.  Institution,
    rank, and total score are metadata/target fields and are not features.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df = df.dropna(subset=["rank"]).sort_values("rank", ascending=True, kind="mergesort")
    elif "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["score"]).sort_values("score", ascending=False, kind="mergesort")

    excluded = {"institution", "rank", "score", "name", "dept"}
    feature_cols = [c for c in df.columns if c not in excluded]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid = X.notna().all(axis=1) & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X = X.loc[valid].copy()

    if X.empty:
        raise ValueError("No valid numeric CSRankings feature rows found.")
    if scale:
        scaler = MinMaxScaler()
        X.loc[:, :] = scaler.fit_transform(X.to_numpy(dtype=float))

    print(f"CSRankings rows: {len(X)}")
    print(f"CSRankings features: {X.shape[1]}")
    return X.to_numpy(dtype=float).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        choices=["nba", "jee", "reco", "reco_thurstone", "reco_mallows",
                                 "csrankings", "online_news"],
                        default="nba",
                        help="Dataset: nba (default), jee, reco, synthetic reco_thurstone/reco_mallows, "
                             "CSRankings, or Online News Popularity")
    parser.add_argument("--jee_csv", type=str, default="./data/jee.csv",
                        help="Path to JEE CSV when --dataset jee")
    parser.add_argument("--jee_scale", type=int, default=0,
                        help="If 1, apply MinMax scaling to JEE features (default 0)")
    parser.add_argument("--reco_csv", type=str, default="./data/reco.csv",
                    help="Path to recommendation synthetic CSV when --dataset reco")
    parser.add_argument("--reco_thurstone_csv", type=str, default="./data/reco_thurstone.csv",
                        help="Path to the Thurstone synthetic CSV when --dataset reco_thurstone")
    parser.add_argument("--reco_mallows_csv", type=str, default="./data/reco_mallows.csv",
                        help="Path to the Mallows synthetic CSV when --dataset reco_mallows")
    parser.add_argument("--csrankings_csv", type=str, default="./data/csrankings_full.csv",
                        help="Path to the CSRankings CSV when --dataset csrankings")
    parser.add_argument("--news_csv", type=str, default="./data/OnlineNewsPopularity/OnlineNewsPopularity.csv",
                        help="Path to OnlineNewsPopularity.csv when --dataset online_news")
    parser.add_argument("--news_feature_set", choices=["all"], default="all",
                        help="Online News feature set (all = 58 predictive features)")
    parser.add_argument("--items", type=int, default=1000)
    parser.add_argument("--dims", type=int, default=None, help="Use the first DIMS features for dimensionality experiments.")
    parser.add_argument("--tk", type=int, default=100)
    parser.add_argument("--randomsuffle", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1000,
                        help="Random samples for initial weights (positive, sum=1)")
    parser.add_argument("--greedy_verbose", type=int, default=0,
                        help="Print greedy refinement progress")

    args = parser.parse_args()

    TK = args.tk
    items = args.items
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "nba":
        data = get_nba_dataset()
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]  # identity top-k
    elif args.dataset == "jee":
        data = get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
        if len(data) < TK:
            raise ValueError(f"JEE dataset has only {len(data)} rows; need at least TK={TK}.")
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]
    elif args.dataset == "reco":
        data = get_reco_dataset(args.reco_csv, scale=False)      # was: get_jee_dataset(args.jee_csv, ...)
        if len(data) < TK:
            raise ValueError(f"Reco dataset has only {len(data)} rows; need at least TK={TK}.")
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]
    elif args.dataset in ("reco_thurstone", "reco_mallows", "csrankings", "online_news"):
        if args.dataset == "reco_thurstone":
            data = get_reco_dataset(args.reco_thurstone_csv, scale=False)
        elif args.dataset == "reco_mallows":
            data = get_reco_dataset(args.reco_mallows_csv, scale=False)
        elif args.dataset == "csrankings":
            data = get_csrankings_dataset(args.csrankings_csv, scale=False)
        else:
            data = get_online_news_dataset(args.news_csv, scale=True, feature_set=args.news_feature_set)
        if items > len(data):
            raise ValueError(f"Requested --items {items}, but dataset has only {len(data)} usable rows.")
        if TK > items:
            raise ValueError(f"--tk ({TK}) cannot exceed --items ({items}).")
        points = data[:items]
        topk_ranking = [[i] for i in range(TK)]

    points_np = np.array(points, dtype=float)
    if args.dims is not None:
        if args.dims < 1 or args.dims > points_np.shape[1]:
            raise ValueError(f"--dims must be between 1 and {points_np.shape[1]}.")
        points_np = points_np[:, :args.dims]
        points = points_np.tolist()

    # Fail early and clearly if any feature is NaN/inf (Gurobi otherwise stops later with
    # "Element k of a double array is Nan or Inf").
    _bad = ~np.isfinite(points_np)
    if _bad.any():
        raise ValueError(
            f"{int(_bad.any(axis=1).sum())} of {len(points_np)} rows contain NaN/inf in feature column(s) "
            f"{np.where(_bad.any(axis=0))[0].tolist()} (0-based). Clean the data file first "
            f"(e.g. run clean_jee.py) or lower --items.")
    ground_truth_topk = list(range(TK))

    
    # ---------- PHASE 1: Random Sampling ----------
    start = time.time()
    best_w, best_err = random_sampling_best(points_np, ground_truth_topk, samples=args.samples, seed=args.seed, verbose=bool(args.verbose))
    t_rs = time.time() - start
    print(f"\n{'='*70}\nPHASE 1: Random Sampling\n{'='*70}")
    print(f"Best sampling error (Kendall-style): {best_err} (time {t_rs:.3f}s)")
    print(f"Best weights (L1-normalized, positive): {best_w}")


    # ---------- PHASE 2: Greedy Refinement ----------
    print(f"\n{'='*70}\nPHASE 2: Greedy Refinement (adjacent swaps + realizability)\n{'='*70}")
    start = time.time()
    refined_w, refined_err = greedy_refine_weights(points_np, best_w, ground_truth_topk, verbose=bool(args.greedy_verbose))
    t_gr = time.time() - start

    # Diagnostics for refined_w
    l2r = np.linalg.norm(refined_w) or 1.0
    alpha_unit_ref = (refined_w / l2r).tolist()
    summary_ref = lppp_diagnostics_counts(points, topk_ranking, alpha_unit_ref, strict=False, epsilon=1e-4)
    total_runtime = t_rs + t_gr
    print(f"total time for LS-INV with {items} and {TK} is {total_runtime:.6f}s (phase 1={t_rs:.6f}s, phase 2={t_gr:.6f}s)")
    print("summary", summary_ref)
   