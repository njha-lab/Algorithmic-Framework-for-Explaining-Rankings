"""
GPP (Grid-based Pairwise Penalty) -- Gurobi implementation in the style of main.py (lppp / dlppp).

How this mirrors main.py
------------------------
* Solver: Gurobi (gurobipy); same Params as lppp():
      OutputFlag=0, Method=2, Presolve=2, Crossover=1, Seed=1, Threads=1, NumericFocus=2
* Data: points = list of lists, topk_ranking = list of tiers [[i], [j, l], ...] (see _normalize_tiers).
* Model (identical to the all-pairs path of lppp, i.e. outside_vs_mintop=False, incremental_*=False):
      alpha = model.addVars(range(dims), lb=0, ub=1)           # here: lb/ub = the current cell
      one slack  e_ij >= 0  per unordered pair (get_err)
      earlier tier -> later tier :  e >= s(t) - s(c) + epsilon
      ties inside a tier         :  e >= s(i) - s(j),  e >= s(j) - s(i)
      top-k -> outsider          :  e >= s(o) - s(t) + epsilon
      objective = sum of slacks
  The ONLY change w.r.t. lppp: the constraint  sum(alpha) == 1  is replaced by the cell box
  l_i <= alpha_i <= u_i.
* Like dlppp(), the model is built ONCE and only the bounds are updated for every cell
  (alpha[i].LB / .UB), then model.optimize() is called again.
* Return value follows lppp:  (raw LP objective of the winning cell, PP error of alpha_unit, alpha_unit).

Why epsilon defaults to 0.0 here
--------------------------------
The (1+eps_approx) guarantee (Theorem 3.3) needs PP to be positively homogeneous, PP(cW) = c PP(W).
A margin epsilon > 0 (as in lppp/dlppp, 1e-4) breaks this.  Keep epsilon=0.0 when you want the
guarantee / a fair comparison with OPT*.  The returned PP error is always computed margin-free.

Algorithm (see Theorem 3.3)
---------------------------
  delta = eps_approx / (sqrt(d) (1 + eps_approx)),  s = delta sqrt(d) < 1
  active cell:  ||l||_2 <= 1 <= ||u||_2             (cell meets the surface ||W||_2 = 1)
  per active cell: solve the LP, normalise  W_hat = W / ||W||_2, score it with the exact PP error
  return the best normalised candidate.
  variant="inside" (ablation only) keeps cells with ||u||_2 <= 1, which is what the old text described.
"""
import itertools
import logging
import math
import time
from itertools import combinations

import gurobipy as gp
import numpy as np

GUROBI_PARAMS = dict(OutputFlag=0, Method=2, Presolve=2, Crossover=1, Seed=1, Threads=1, NumericFocus=2)


# ----------------------------------------------------------------------------
# helpers copied from / consistent with main.py
# ----------------------------------------------------------------------------
def _normalize_tiers(topk_ranking):
    def _is_intlike(x):
        return isinstance(x, int) or isinstance(x, np.integer)
    if isinstance(topk_ranking, set):
        return [list(int(i) for i in topk_ranking)]
    elif hasattr(topk_ranking, "__iter__") and all(_is_intlike(x) for x in topk_ranking):
        return [[int(i)] for i in topk_ranking]
    else:
        return [[int(i) for i in tier] for tier in topk_ranking]


def pair_matrix(points, topk_ranking):
    """Rows a_p with PP(W) = sum_p max(0, a_p.W); the same pairs that carry a slack in the model."""
    X = np.asarray(points, dtype=float)
    n = len(X)
    tiers = _normalize_tiers(topk_ranking)
    ordered = [i for t in tiers for i in t]
    outsiders = sorted(set(range(n)) - set(ordered))
    rows, seen = [], []
    for tier in tiers:
        for c in seen:
            for t in tier:
                rows.append(X[t] - X[c])                       # e >= s(t) - s(c)
        for i, j in combinations(tier, 2):
            rows.append(X[i] - X[j]); rows.append(X[j] - X[i])  # ties: |s(i) - s(j)|
        seen.extend(tier)
    for t in ordered:
        for o in outsiders:
            rows.append(X[o] - X[t])                           # e >= s(o) - s(t)
    return np.array(rows) if rows else np.zeros((0, X.shape[1]))


def pp_error(points, topk_ranking, alpha, A=None):
    """Margin-free PP error of alpha / ||alpha||_2 (paper Eq. 1, ties as in Eq. 5)."""
    A = pair_matrix(points, topk_ranking) if A is None else A
    W = np.asarray(alpha, dtype=float)
    return float(np.maximum(A @ (W / np.linalg.norm(W)), 0.0).sum())


# ----------------------------------------------------------------------------
# model: lppp's all-pairs LP, with the box in place of sum(alpha) == 1
# ----------------------------------------------------------------------------
def _build_model(points, tiers, epsilon, params):
    n, dims = len(points), len(points[0])
    model = gp.Model()
    for k, v in params.items():
        setattr(model.Params, k, v)

    ordered_topk = [i for tier in tiers for i in tier]
    outsiders = sorted(set(range(n)) - set(ordered_topk))

    alpha = model.addVars(range(dims), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

    def score_expr(i):
        return gp.quicksum(alpha[d] * points[i][d] for d in range(dims))

    error_vars = {}
    def get_err(i, j):
        if i > j: i, j = j, i
        if i not in error_vars:
            error_vars[i] = {}
        if j not in error_vars[i]:
            error_vars[i][j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_{i}_{j}")
        return error_vars[i][j]

    obj = gp.LinExpr()
    curr_proc = []
    for tier in tiers:
        for c in curr_proc:
            for t in tier:
                e = get_err(c, t)
                model.addConstr(e >= score_expr(t) - score_expr(c) + epsilon)
                obj += e
        curr_proc.extend(tier)
        if len(tier) > 1:
            for i, j in combinations(tier, 2):
                e = get_err(i, j)
                model.addConstr(e >= score_expr(i) - score_expr(j))
                model.addConstr(e >= score_expr(j) - score_expr(i))
                obj += e
    for t in ordered_topk:
        for o in outsiders:
            e = get_err(t, o)
            model.addConstr(e >= score_expr(o) - score_expr(t) + epsilon)
            obj += e
    model.setObjective(obj, gp.GRB.MINIMIZE)
    return model, alpha


# ----------------------------------------------------------------------------
# grid cells
# ----------------------------------------------------------------------------
def _active_cells(d, delta, M, tol=1e-12):
    """
    Cells H(l), l = delta*c, c in {0..M-1}^d, with ||l||_2 <= 1 <= ||l + delta*1||_2 (cell meets the surface).
    O(d * M^(d-1)): scan the first d-1 coordinates; the admissible last indices form an interval.
    Tolerances only ever ADD borderline cells (harmless: GPP may process any superset of active cells).
    """
    for prefix in itertools.product(range(M), repeat=d - 1):
        l2 = sum((c * delta) ** 2 for c in prefix)
        if l2 > 1 + tol:
            continue
        u2 = sum(((c + 1) * delta) ** 2 for c in prefix)
        hi = math.floor(math.sqrt(max(0.0, 1 + tol - l2)) / delta + tol)
        need = 1 - tol - u2
        lo = 0 if need <= 0 else math.ceil(math.sqrt(need) / delta - tol) - 1
        for c in range(max(0, lo), min(M - 1, hi) + 1):
            yield prefix + (c,)


def _inside_cells(d, delta, M, tol=1e-12):
    for c in itertools.product(range(M), repeat=d):
        if sum(((ci + 1) * delta) ** 2 for ci in c) <= 1 + tol:
            yield c


def count_active_cells(d, eps_approx):
    delta = eps_approx / (math.sqrt(d) * (1 + eps_approx))
    return sum(1 for _ in _active_cells(d, delta, math.ceil(1 / delta - 1e-12)))


def lp_count_bound(d, eps_approx):
    """Proved upper bound on the number of active cells / LPs (Theorem 3.3)."""
    return 2 * math.sqrt(d / math.pi) * (math.pi * math.e / 2) ** (d / 2) * ((1 + 2 * eps_approx) / eps_approx) ** (d - 1)


# ----------------------------------------------------------------------------
# GPP
# ----------------------------------------------------------------------------
def gpp(points, topk_ranking, *,
        eps_approx: float = None,
        delta: float = None,
        strict: bool = False,
        epsilon: float = 0.0,
        variant: str = "active",
        params: dict = None,
        stats: dict = None,
        verbose: bool = False):
    """
    Returns (raw LP objective of the winning cell, PP error of alpha_unit, alpha_unit)
    or (None, None, None) if no cell could be solved to optimality.
    `stats` (optional dict) is filled with: delta, s, M, n_cells, n_lp, n_failed, min_norm, time.
    """
    if strict:
        raise NotImplementedError("GPP is defined for the soft (PP) objective; use strict=False.")
    t0 = time.time()
    n, dims = len(points), len(points[0])
    if delta is None:
        assert eps_approx is not None and eps_approx > 0, "give eps_approx > 0 (or delta)"
        delta = eps_approx / (math.sqrt(dims) * (1.0 + eps_approx))
    s = delta * math.sqrt(dims)
    assert s < 1.0, "cell diameter delta*sqrt(d) must be < 1"
    M = math.ceil(1.0 / delta - 1e-12)

    cells_iter = _active_cells(dims, delta, M) if variant == "active" else _inside_cells(dims, delta, M)
    cells = list(cells_iter)

    tiers = _normalize_tiers(topk_ranking)
    model, alpha = _build_model(points, tiers, epsilon, {**GUROBI_PARAMS, **(params or {})})
    A = pair_matrix(points, topk_ranking)

    best_raw, best_err, best_alpha = math.inf, math.inf, None
    n_lp = n_failed = 0
    min_norm = math.inf
    for c in cells:
        for i in range(dims):                         # like dlppp(): update the model, do not rebuild it
            alpha[i].LB = c[i] * delta
            alpha[i].UB = c[i] * delta + delta
        model.optimize()
        n_lp += 1
        if model.Status != gp.GRB.OPTIMAL:            # the guarantee needs an optimal LP solution
            n_failed += 1
            logging.info(f"Failed with status {model.Status}")
            continue
        W = np.array([alpha[i].X for i in range(dims)])
        nW = float(np.linalg.norm(W))
        min_norm = min(min_norm, nW)
        if nW < 1e-12:                                # only possible for cells containing the origin
            continue
        W_hat = W / nW                                # projection onto ||W||_2 = 1
        err = float(np.maximum(A @ W_hat, 0.0).sum()) # exact PP error of the normalised candidate
        if err < best_err:
            best_raw, best_err, best_alpha = model.ObjVal, err, W_hat
        if verbose:
            print(f"cell {c}: ||W||={nW:.4f} LP obj={model.ObjVal:.6g} PP(W_hat)={err:.6g}")

    if stats is not None:
        stats.update(delta=delta, s=s, M=M, n_cells=len(cells), n_lp=n_lp, n_failed=n_failed,
                     min_norm=min_norm, time=time.time() - t0)
    logging.info(f"GPP n={n} d={dims} delta={delta:.5f} cells={len(cells)} failed={n_failed} "
                 f"best_err={best_err} TIME={time.time() - t0:.2f}")
    if best_alpha is None:
        return None, None, None
    return best_raw, best_err, best_alpha.tolist()


# ----------------------------------------------------------------------------
# CLI (flags follow main.py)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["nba", "jee", "reco", "reco_thurstone", "reco_mallows", "csrankings", "online_news"], default="nba")
    parser.add_argument("--jee_csv", type=str, default="./data/jee.csv")
    parser.add_argument("--reco_csv", type=str, default="./data/reco.csv")
    parser.add_argument("--reco_thurstone_csv", type=str, default="./data/reco_thurstone.csv")
    parser.add_argument("--reco_mallows_csv", type=str, default="./data/reco_mallows.csv")
    parser.add_argument("--csrankings_csv", type=str, default="./data/csrankings_full.csv")
    parser.add_argument("--news_csv", type=str, default="./data/OnlineNewsPopularity/OnlineNewsPopularity.csv")
    parser.add_argument("--jee_scale", type=int, default=1)
    parser.add_argument("--items", type=int, default=30)
    parser.add_argument("--tk", type=int, default=8)
    parser.add_argument("--dims", type=int, default=3, help="use the first DIMS features (GPP is exponential in d)")
    parser.add_argument("--eps", type=float, default=0.5, help="approximation parameter eps (guarantee 1+eps)")
    parser.add_argument("--epsilon", type=float, default=0.0, help="constraint margin; 0 keeps the guarantee")
    parser.add_argument("--variant", choices=["active", "inside"], default="active")
    parser.add_argument("--compare_lppp", type=int, default=0, help="also run lppp from main.py (all-pairs LP)")
    parser.add_argument("--opt", type=int, default=0, help="exact OPT* via exact_opt.py (Gurobi, small instances)")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    import main as M                                   # your main.py (datasets, lppp)
    M.args = args                                      # lppp_diagnostics_counts reads a global `args`
    if args.dataset == "nba":
        data = M.get_nba_dataset()
    elif args.dataset == "jee":
        # Uses main.py's shared JEE loader: stable descending mark order
        # is applied before feature construction; the source CSV is untouched.
        data = M.get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
    elif args.dataset == "reco":
        data = M.get_reco_dataset(args.reco_csv, scale=False)
    elif args.dataset == "reco_thurstone":
        data = M.get_reco_dataset(args.reco_thurstone_csv, scale=False)
    elif args.dataset == "reco_mallows":
        data = M.get_reco_dataset(args.reco_mallows_csv, scale=False)
    elif args.dataset == "csrankings":
        data = M.get_csrankings_dataset(args.csrankings_csv, scale=False)
    else:
        data = M.get_online_news_dataset(args.news_csv, scale=True, feature_set="all")
    if args.dims < 1 or args.dims > len(data[0]):
        raise ValueError(f"--dims must be between 1 and {len(data[0])}.")
    points = [row[:args.dims] for row in data[:args.items]]
    TK = args.tk
    topk_ranking = [[i] for i in range(TK)]

    stats = {}
    if args.opt:
        A = pair_matrix(points, topk_ranking)
        from exact_opt import compute_opt_star
        o = compute_opt_star(A, time_limit=300)
    for eps_val in [args.eps]:
        start = time.time()
        raw, err, alpha_unit = gpp(points, topk_ranking, eps_approx=eps_val, epsilon=args.epsilon,
                               variant=args.variant, stats=stats, verbose=bool(args.verbose))
        print(" -------------------------------------")
      
        print(alpha_unit)
        print(f"Epsilon value for this interation : {eps_val}")
        print(f"total time for gpp with {args.items} and {TK} is {time.time() - start} and opt is {raw}")
        if err is None:
            print(f"GPP failed for eps={eps_val}")
            continue

        if args.opt:
            A = pair_matrix(points, topk_ranking)
            from exact_opt import compute_opt_star
            o = compute_opt_star(A, time_limit=300)
            print(f"exact OPT* in [{o['lb']}, {o['ub']}];  GPP/OPT* = {err / o['lb'] if o['lb'] > 1e-12 else float('nan')}")

        print(f"PP error of alpha_unit = {err}   stats = {stats}")
        logging.info(f"GPP {args.items} {TK} {(raw, err, alpha_unit)} {stats}")
        logging.info(f"GPP {args.items} {TK} {(raw, err, alpha_unit)} {stats}")
        print(" -------------------------------------")

    A = pair_matrix(points, topk_ranking)
    if args.compare_lppp:
        out = M.lppp(points, topk_ranking, outside_vs_mintop=False, incremental_outside=False,
                     incremental_topk=False, earlybreak=False, strict=False, epsilon=args.epsilon)
        print(f"lppp (all pairs): PP error of alpha_unit = {pp_error(points, topk_ranking, out[2], A)}")
    
