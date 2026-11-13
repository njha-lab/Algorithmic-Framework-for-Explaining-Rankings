"""
Exact / certified OPT* for the Pairwise-Penalty (PP) objective on small instances.

PP(W) = sum_p max(0, a_p . W)         (one row a_p per ordered pair, see build_pair_matrix)
OPT*  = min  PP(W)   s.t.  ||W||_2 = 1,  W >= 0

Key fact: PP is convex, piecewise linear and positively homogeneous, PP(cW) = c PP(W).  Hence
  (i)  ||W||_2 = 1 can be replaced by ||W||_2 >= 1 without changing the optimum, and
  (ii) PP(W)/||W||_2 is the PP of the L2-normalised weights, for any W != 0.

Methods
-------
opt_star_gurobi  : global spatial branch-and-bound (NonConvex=2). Returns a certified lower
                   bound (ObjBound) even on timeout.
lp_l1            : the paper's L1 LP (Eq. 5, full pairs). Always LP* <= OPT*.
"""
from __future__ import annotations

import itertools
import math
import time
import csv
import argparse

import numpy as np


# ----------------------------------------------------------------------------
# Problem data
# ----------------------------------------------------------------------------
def normalize_tiers(topk_ranking):
    """Accepts [[i],[j,l],...] (tiers), [i,j,l,...] (strict order) or a set (one tie tier)."""
    if isinstance(topk_ranking, (set, frozenset)):
        return [sorted(int(i) for i in topk_ranking)]
    tiers = []
    for t in topk_ranking:
        tiers.append([int(t)] if np.isscalar(t) else [int(i) for i in t])
    return tiers


#def build_pair_matrix(points, topk_ranking):
    """
    Rows a_p such that PP(W) = sum_p max(0, a_p . W)  (paper Eq. 1, no epsilon margin):
      * r in an earlier tier than s (both top-k):  a = x_s - x_r
      * ties inside a tier:                        a = x_i - x_j  and  x_j - x_i  (-> |.|)
      * r in top-k, o outside top-k:               a = x_o - x_r
    """
 #   X = np.asarray(points, dtype=float)
  #  n, d = X.shape
   # tiers = normalize_tiers(topk_ranking)
    #topk = [i for t in tiers for i in t]
    #inside = set(topk)
    #rows = []
    #for a in range(len(tiers)):
     #   for b in range(a + 1, len(tiers)):
      #      for r in tiers[a]:
       #         for s in tiers[b]:
        #            rows.append(X[s] - X[r])
    #for t in tiers:
     #   for i, j in itertools.combinations(t, 2):
      #      rows.append(X[i] - X[j])
       #     rows.append(X[j] - X[i])
   # outside = np.array([o for o in range(n) if o not in inside], dtype=int)
    #if len(outside) and len(topk):
     #   diff = X[outside][None, :, :] - X[np.array(topk)][:, None, :]
      #  rows.extend(diff.reshape(-1, d))
    #return np.array(rows, dtype=float) if rows else np.zeros((0, d))

def build_pair_matrix(points, topk_ranking, mode="topk"):
    """
    Build pairwise-difference matrix for the PP objective.

    mode="topk":
        Uses the top-k formulation:
          1. Earlier top-k tuple vs later top-k tuple
          2. Ties within top-k
          3. Top-k tuple vs every outside tuple

    mode="full":
        Uses every pair in the complete ranking:
          tuple i vs tuple j for all i < j.

    For every row a = x_j - x_i,
        max(0, a @ W)
    is the PP penalty when j is incorrectly ranked above i.
    """
    X = np.asarray(points, dtype=float)
    n, d = X.shape

    # ---------------------------------------------------------
    # FULL-RANKING MODE
    # ---------------------------------------------------------
    if mode == "full":
        rows = []

        for i in range(n):
            for j in range(i + 1, n):
                rows.append(X[j] - X[i])

        return (
            np.array(rows, dtype=float)
            if rows
            else np.zeros((0, d))
        )

    # ---------------------------------------------------------
    # TOP-K MODE
    # ---------------------------------------------------------
    if mode != "topk":
        raise ValueError(
            f"Unknown mode '{mode}'. Use 'topk' or 'full'."
        )

    tiers = normalize_tiers(topk_ranking)

    topk = [i for t in tiers for i in t]
    inside = set(topk)

    rows = []

    # Earlier tier vs later tier
    for a in range(len(tiers)):
        for b in range(a + 1, len(tiers)):
            for r in tiers[a]:
                for s in tiers[b]:
                    rows.append(X[s] - X[r])

    # Ties inside a tier
    for t in tiers:
        for i, j in itertools.combinations(t, 2):
            rows.append(X[i] - X[j])
            rows.append(X[j] - X[i])

    # Top-k vs outside
    outside = np.array(
        [o for o in range(n) if o not in inside],
        dtype=int
    )

    if len(outside) and len(topk):
        diff = (
            X[outside][None, :, :]
            - X[np.array(topk)][:, None, :]
        )
        rows.extend(diff.reshape(-1, d))

    return (
        np.array(rows, dtype=float)
        if rows
        else np.zeros((0, d))
    )

def pp(A, W):
    """Raw PP of W (not normalised)."""
    return float(np.maximum(A @ np.asarray(W, float), 0.0).sum())


def pp_l2(A, W):
    """PP of W / ||W||_2.  This is the quantity to compare with OPT*."""
    W = np.asarray(W, float)
    return pp(A, W) / float(np.linalg.norm(W))


# ----------------------------------------------------------------------------
# Lower bound: the paper's L1 LP
# ----------------------------------------------------------------------------
def lp_l1(A):
    """
    min sum_p s_p  s.t. s_p >= a_p.W, s_p >= 0, W >= 0, sum W = 1.   (Eq. 5, all pairs)
    Returns (LP*, W_lp).  LP* <= OPT*  and  PP(W_lp/||W_lp||_2) <= sqrt(d) * OPT*.
    Proof of the first: W*/||W*||_1 is L1-feasible, ||W*||_1 >= ||W*||_2 = 1, PP homogeneous.
    """
    import scipy.sparse as sp
    from scipy.optimize import linprog

    m, d = A.shape
    c = np.r_[np.zeros(d), np.ones(m)]
    A_ub = sp.hstack([sp.csr_matrix(A), -sp.identity(m, format="csr")], format="csr")
    A_eq = np.r_[np.ones(d), np.zeros(m)][None, :]
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (d + m), method="highs")
    assert res.status == 0, res.message
    return float(res.fun), res.x[:d]


# ----------------------------------------------------------------------------
# Exact method: Gurobi global solver  (certified bound)
# ----------------------------------------------------------------------------
def opt_star_gurobi(A, W_start=None, time_limit=600.0, mip_gap=1e-6, threads=1,
                    verbose=False, valid_cuts=True):
    """
    min sum s_p   s.t.  s_p >= a_p.w,  s_p >= 0,  sum w_k^2 = 1,  0 <= w <= 1.
    Needs Params.NonConvex = 2.  Valid, redundant cuts 1 <= sum w <= sqrt(d) (since
    ||w||_2 <= ||w||_1 <= sqrt(d)||w||_2) tighten the relaxation.

    Returns dict(value, bound, gap, W, status, time).
      value : PP of the returned W/||W||_2   (an upper bound on OPT*, recomputed exactly)
      bound : Gurobi ObjBound                (a lower bound on OPT*, up to feasibility tol)
    """
    import gurobipy as gp
    from gurobipy import GRB

    m, d = A.shape
    mdl = gp.Model("opt_star")
    mdl.Params.OutputFlag = int(verbose)
    mdl.Params.NonConvex = 2
    mdl.Params.TimeLimit = time_limit
    mdl.Params.MIPGap = mip_gap
    mdl.Params.Threads = threads

    w = mdl.addVars(d, lb=0.0, ub=1.0, name="w")
    s = mdl.addVars(m, lb=0.0, name="s")
    for p in range(m):
        mdl.addConstr(s[p] >= gp.quicksum(float(A[p, k]) * w[k] for k in range(d) if A[p, k] != 0.0))
    mdl.addConstr(gp.quicksum(w[k] * w[k] for k in range(d)) == 1.0, name="sphere")
    if valid_cuts:
        mdl.addConstr(gp.quicksum(w[k] for k in range(d)) >= 1.0)
        mdl.addConstr(gp.quicksum(w[k] for k in range(d)) <= math.sqrt(d))
    mdl.setObjective(gp.quicksum(s[p] for p in range(m)), GRB.MINIMIZE)

    if W_start is not None:                                     # warm start (e.g. LP-PP output)
        W0 = np.asarray(W_start, float)
        W0 = W0 / np.linalg.norm(W0)
        for k in range(d):
            w[k].Start = float(W0[k])
        sc = np.maximum(A @ W0, 0.0)
        for p in range(m):
            s[p].Start = float(sc[p])

    t0 = time.time()
    mdl.optimize()
    out = dict(status=mdl.Status, time=time.time() - t0, W=None, value=None, bound=None, gap=None)
    if mdl.SolCount > 0:
        Wv = np.array([w[k].X for k in range(d)])
        Wv = np.clip(Wv, 0.0, None)
        Wv /= np.linalg.norm(Wv)
        out["W"] = Wv
        out["value"] = pp(A, Wv)                                # recomputed, so a true upper bound
        out["bound"] = max(0.0, float(mdl.ObjBound))
        out["gap"] = (out["value"] - out["bound"]) / max(abs(out["value"]), 1e-12)
    return out


# ----------------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------------
def compute_opt_star(A, **gurobi_kw):
    """
    Returns dict(lb, ub, W, exact, time, lp_l1, gap).  OPT* in [lb, ub]; exact iff lb ~= ub.
    """
    lp_val, W_lp = lp_l1(A)
    r = opt_star_gurobi(A, W_start=W_lp, **gurobi_kw)
    lb = max(r["bound"], lp_val)                                # LP* is also a valid lower bound
    ub = r["value"]
    return dict(lb=lb, ub=ub, W=r["W"], exact=(ub - lb) <= 1e-6 * max(1.0, ub),
                method="gurobi", time=r["time"], lp_l1=lp_val, gap=r["gap"])


# ----------------------------------------------------------------------------
# Real-input runner: paper running example or a prefix of a project dataset
# ----------------------------------------------------------------------------
PAPER_RUNNING_EXAMPLE = np.array([
    [75, 95, 65],  # C6
    [82, 85, 90],  # C1
    [88, 92, 75],  # C2
    [90, 80, 70],  # C4
    [70, 78, 95],  # C5
    [92, 70, 80],  # C8
    [85, 88, 60],  # C3
    [60, 65, 85],  # C7
], dtype=float)
PAPER_RUNNING_EXAMPLE_NAMES = ["C6", "C1", "C2", "C4", "C5", "C8", "C3", "C7"]


def load_points(args):
    """Load a real dataset using the same loaders/feature processing as main.py."""
    if args.dataset == "running_example":
        return PAPER_RUNNING_EXAMPLE.copy(), "paper_running_example"

    if args.dataset == "csv":
        import pandas as pd
        if not args.csv_path:
            raise ValueError("For --dataset csv, provide --csv_path PATH.")
        df = pd.read_csv(args.csv_path, low_memory=False)
        # Generic CSV mode uses numeric columns only; users should remove IDs,
        # target labels, and other non-feature columns before running this mode.
        data = df.select_dtypes(include=[np.number]).copy()
        if data.empty:
            raise ValueError(f"No numeric columns found in {args.csv_path}.")
        data = data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if data.empty:
            raise ValueError(f"No complete finite numeric rows found in {args.csv_path}.")
        return data.to_numpy(dtype=float), str(args.csv_path)

    # Reuse the project's own dataset loaders so preprocessing matches main.py.
    # This import is delayed so the paper running example can run independently
    # of the main.py dataset dependencies.
    try:
        import main as project_main
    except Exception as exc:
        raise RuntimeError(
            "Could not import main.py dataset loaders. Keep exact_opt.py beside "
            "main.py and install the project's dependencies."
        ) from exc

    if args.dataset == "nba":
        data = project_main.get_nba_dataset()
        source = "./data/per.csv (via main.get_nba_dataset)"
    elif args.dataset == "jee":
        # Shared loader in main.py stably sorts by mark descending before
        # constructing the feature matrix; raw jee.csv remains unchanged.
        data = project_main.get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
        source = args.jee_csv
    elif args.dataset == "reco":
        data = project_main.get_reco_dataset(args.reco_csv, scale=False)
        source = args.reco_csv
    elif args.dataset == "online_news":
        data = project_main.get_online_news_dataset(
            args.news_csv, scale=True, feature_set=args.news_feature_set
        )
        source = args.news_csv
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Dataset {args.dataset!r} did not produce a non-empty 2D feature matrix.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Dataset {args.dataset!r} contains NaN or infinite feature values after loading.")
    return arr, source


def run_one_experiment(args):
    points_all, source = load_points(args)
    n_available = len(points_all)
    default_items = n_available if args.dataset == "running_example" else min(20, n_available)
    n = default_items if args.items is None else args.items
    if n < 2:
        raise ValueError("--items must be at least 2.")
    if n > n_available:
        raise ValueError(f"Requested --items {n}, but dataset has only {n_available} usable rows.")
    points = np.asarray(points_all[:n], dtype=float)
    d = points.shape[1]

    if args.mode is None:
        mode = "full" if args.dataset == "running_example" else "topk"
    else:
        mode = args.mode

    if args.topk_indices:
        try:
            topk_indices = [int(v.strip()) for v in args.topk_indices.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError("--topk_indices must be comma-separated zero-based row indices, e.g. 0,1,2") from exc
        if not topk_indices:
            raise ValueError("--topk_indices was supplied but contained no indices.")
        if len(set(topk_indices)) != len(topk_indices):
            raise ValueError("--topk_indices cannot contain duplicate indices.")
        if any(i < 0 or i >= n for i in topk_indices):
            raise ValueError(f"Every --topk_indices value must be between 0 and {n - 1}.")
        topk_ranking = [[i] for i in topk_indices]
    else:
        k = args.tk
        if k < 1 or k > n:
            raise ValueError(f"--tk must be between 1 and --items ({n}); received {k}.")
        topk_ranking = [[i] for i in range(k)]
        topk_indices = list(range(k))

    A = build_pair_matrix(points, topk_ranking, mode=mode)
    m = len(A)
    if m == 0:
        raise ValueError("The selected ranking and mode generated no pairwise comparisons.")

    print("=" * 72)
    print("EXACT OPT* EXPERIMENT")
    print("=" * 72)
    print(f"Dataset/source : {args.dataset} ({source})")
    print(f"Input size     : n={n}, d={d}")
    print(f"Ranking mode   : {mode}")
    print(f"Top-k indices  : {topk_indices} (zero-based row indices)")
    print(f"Pair comparisons: m={m}")
    if args.dataset == "running_example":
        print("Paper tuples   : " + ", ".join(
            f"{PAPER_RUNNING_EXAMPLE_NAMES[i]}={tuple(int(v) for v in points[i])}" for i in range(n)
        ))
    print("\nSolving the L1 LP lower bound...")
    lp_val, W_lp = lp_l1(A)
    pp_lp = pp_l2(A, W_lp)
    print(f"LP* lower bound                 : {lp_val:.10f}")
    print(f"PP of normalized LP weights     : {pp_lp:.10f}")
    print("LP weights (L1-normalized)      : " + np.array2string(W_lp, precision=8))

    print("\nSolving the unit-L2 Gurobi model...")
    g = opt_star_gurobi(
        A,
        W_start=W_lp,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        verbose=args.verbose,
        valid_cuts=not args.no_valid_cuts,
    )
    if g["value"] is None or g["bound"] is None or g["W"] is None:
        raise RuntimeError(
            f"Gurobi returned no feasible solution (status={g['status']}). "
            "Increase --time_limit or inspect the solver log. No result row was written."
        )

    lower_bound = max(lp_val, g["bound"])
    upper_bound = g["value"]
    ratio = pp_lp / lower_bound if lower_bound > 1e-12 else float("nan")
    bound_gap = upper_bound - lower_bound
    certified_to_tolerance = bound_gap <= args.cert_tol * max(1.0, abs(upper_bound))
    inequalities_ok = (
        # The LP and Gurobi bounds are both lower bounds; neither must be
        # stronger than the other. Their maximum must not exceed the feasible UB.
        lower_bound <= g["value"] + args.check_tol
        and g["bound"] <= g["value"] + args.check_tol
        and g["value"] <= pp_lp + args.check_tol
        and pp_lp <= math.sqrt(d) * lower_bound + args.check_tol
    )

    print("\nRESULTS")
    print(f"Gurobi candidate PP / upper bound : {upper_bound:.10f}")
    print(f"Gurobi objective lower bound      : {g['bound']:.10f}")
    print(f"Combined lower bound              : {lower_bound:.10f}")
    print(f"OPT* interval                     : [{lower_bound:.10f}, {upper_bound:.10f}]")
    print(f"Interval width                    : {bound_gap:.10g}")
    print(f"Relative Gurobi gap               : {g['gap']:.10g}")
    print(f"PP(LP-normalized weights)         : {pp_lp:.10f}")
    print(f"Approximation ratio PP(LP)/LB (certified ratio upper bound)   : {ratio:.8f} (target <= sqrt(d)={math.sqrt(d):.8f})")
    print(f"Gurobi status code                : {g['status']}")
    print(f"Gurobi solve time                 : {g['time']:.4f} seconds")
    print(f"Inequality checks                 : {'PASS' if inequalities_ok else 'CHECK/FAIL'}")
    print(f"Certified to requested tolerance  : {'YES' if certified_to_tolerance else 'NO'}")
    print("Optimal unit-L2 weights           : " + np.array2string(g["W"], precision=10))

    # Save a one-row summary and a separate, readable weight file. In the
    # screen wrapper these paths are relative to that run's unique log folder.
    output_path = args.output_csv
    weight_path = args.weights_csv
    summary_fields = [
        "dataset", "source", "mode", "n", "k", "d", "m", "lp_star",
        "pp_lp_normalized", "opt_star_upper_bound", "gurobi_lower_bound",
        "combined_lower_bound", "interval_width", "gurobi_relative_gap",
        "approximation_ratio", "gurobi_status", "solve_time_seconds",
        "inequality_checks", "certified_to_tolerance", "weights"
    ]
    summary_row = {
        "dataset": args.dataset,
        "source": source,
        "mode": mode,
        "n": n,
        "k": len(topk_indices),
        "d": d,
        "m": m,
        "lp_star": lp_val,
        "pp_lp_normalized": pp_lp,
        "opt_star_upper_bound": upper_bound,
        "gurobi_lower_bound": g["bound"],
        "combined_lower_bound": lower_bound,
        "interval_width": bound_gap,
        "gurobi_relative_gap": g["gap"],
        "approximation_ratio": ratio,
        "gurobi_status": g["status"],
        "solve_time_seconds": g["time"],
        "inequality_checks": "PASS" if inequalities_ok else "CHECK/FAIL",
        "certified_to_tolerance": "YES" if certified_to_tolerance else "NO",
        "weights": ";".join(f"{v:.12g}" for v in g["W"]),
    }
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)
        f.flush()
    with open(weight_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "weight"])
        for j, value in enumerate(g["W"]):
            writer.writerow([j, f"{value:.12g}"])
        f.flush()
    print(f"Summary saved to : {output_path}")
    print(f"Weights saved to : {weight_path}")
    return summary_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute/certify OPT* for the paper running example or a small prefix "
            "of one of the project's datasets. No random instances are generated."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["running_example", "nba", "jee", "reco", "online_news", "csv"],
        default="running_example",
        help="Input source (default: the paper's 8-tuple running example).",
    )
    parser.add_argument("--items", type=int, default=None,
                        help="Use the first n rows; defaults to all 8 running-example rows or 20 rows for other datasets.")
    parser.add_argument("--tk", type=int, default=3,
                        help="Top-k size when --topk_indices is not supplied (default: 3).")
    parser.add_argument("--mode", choices=["topk", "full"], default=None,
                        help="Comparison mode. Default is full for running_example, topk for other datasets.")
    parser.add_argument("--topk_indices", type=str, default=None,
                        help="Optional zero-based strict ranking indices, e.g. 0,1,2. Defaults to first --tk rows.")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Numeric CSV file to use with --dataset csv; row order is treated as reference order.")
    parser.add_argument("--jee_csv", type=str, default="./data/jee.csv")
    parser.add_argument("--jee_scale", type=int, choices=[0, 1], default=1,
                        help="Use the same JEE feature scaling as main.py (default: 1).")
    parser.add_argument("--reco_csv", type=str, default="./data/reco.csv")
    parser.add_argument("--news_csv", type=str, default="./data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    parser.add_argument("--news_feature_set", choices=["all"], default="all")
    parser.add_argument("--time_limit", type=float, default=60.0,
                        help="Gurobi time limit in seconds for this instance (default: 60).")
    parser.add_argument("--mip_gap", type=float, default=1e-6,
                        help="Gurobi relative MIP gap target (default: 1e-6).")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=0,
                        help="Show detailed Gurobi log when set to 1.")
    parser.add_argument("--no_valid_cuts", action="store_true",
                        help="Disable the redundant L1-norm strengthening constraints.")
    parser.add_argument("--cert_tol", type=float, default=1e-6,
                        help="Tolerance for deciding whether the OPT* interval is sufficiently narrow.")
    parser.add_argument("--check_tol", type=float, default=1e-6,
                        help="Numerical tolerance for theoretical inequality checks.")
    parser.add_argument("--output_csv", default="exact_opt_results.csv",
                        help="One-row summary CSV path (default: exact_opt_results.csv).")
    parser.add_argument("--weights_csv", default="exact_opt_weights.csv",
                        help="Learned weights CSV path (default: exact_opt_weights.csv).")
    cli_args = parser.parse_args()
    run_one_experiment(cli_args)
