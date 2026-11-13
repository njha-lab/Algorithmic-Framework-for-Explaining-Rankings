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

"""
# Configure basic logging to a file
logging.basicConfig(
    filename='output.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
"""
# Configure logging to a file
file_handler = logging.FileHandler('app.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

"""
# Log some messages
logging.info('Application started successfully.')
logging.warning('Potential issue detected.')
try:
    result = 10 / 0
except ZeroDivisionError:
    logging.error('Attempted division by zero!', exc_info=True) # exc_info=True adds exception details
logging.critical('System shutdown initiated.')

"""
import math
import itertools
import gurobipy as gp
from itertools import combinations

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
        "pp_l2": float,                        # pp scaled by ||alpha||_2
        "topk_inversions_count": int,          # earlier (better) scored worse than later (worse)
        "outside_pair_violations": int,        # count of (outsider, topk) with s[o]-s[t]+margin > tol
        "outsiders_with_any_violation": int,   # number of distinct outsiders with any such violation
        "max_abs_rank_diff": int,              # max |pos_score - pos_given| over top-k items
        "sum_abs_rank_diff": int               # sum |pos_score - pos_given| over top-k items
      }
    """
    n, dims = len(points), len(points[0])
    assert len(alpha_unit) == dims, "alpha dimension mismatch"

    # scores under the provided alpha
    scores = [sum(alpha_unit[d]*points[i][d] for d in range(dims)) for i in range(n)]

    tiers = _normalize_tiers(topk_ranking)
    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = [i for i in range(n) if i not in topk_set]

    margin = 0.0 if strict else epsilon

    # ----- existing violation metrics (unchanged logic) -----
    count_pp = 0.0
    inversions = 0
    seen = []
    for tier in tiers:
        for t in tier:               # later/worse
            for c in seen:           # earlier/better
                gap = scores[t] - scores[c]
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
            if gap + margin > tol:
                pair_viol += 1
                count_pp += gap
                violated = True
        if violated:
            outsider_any.add(o)

    # ----- L2 scaling for pp -----
    # robust even if a non-unit alpha is passed
    l2 = math.sqrt(sum(v*v for v in alpha_unit)) or 1.0
    pp_l2 = count_pp / l2

    # ----- ranking discrepancy between score ranking and given top-k order -----
    # score-based ranking over all items, descending score
    score_ranking = sorted(range(n), key=lambda i: scores[i], reverse=True)
    pos_score = {i: r for r, i in enumerate(score_ranking)}
    pos_given = {i: r for r, i in enumerate(ordered_topk)}  # minimal, expands ties as listed
    # compute discrepancies only over top-k items
    abs_diffs = [abs(pos_score[i] - pos_given[i]) for i in ordered_topk]
    max_abs_rank_diff = max(abs_diffs) if abs_diffs else 0
    sum_abs_rank_diff = int(sum(abs_diffs)) if abs_diffs else 0

    return {
        "pp": count_pp,
        "topk_inversions_count": inversions,
        "outside_pair_violations": pair_viol,
        "outsiders_with_any_violation": len(outsider_any),
        "max_abs_rank_diff": max_abs_rank_diff,
        "sum_abs_rank_diff": sum_abs_rank_diff,
    }

# =======================
# Unified LPPP function
# =======================
def lppp(points, topk_ranking, *,
         outside_vs_mintop: bool,
         incremental_outside: bool,
         incremental_topk: bool,
         earlybreak: bool,
         strict: bool = False,
         epsilon: float = 1e-4,
         max_rounds: int = 50,
         tol: float = 1e-12,
         outside_mode: str = "aggregate_exact"):
    """

      outside_vs_mintop=True
          Compare outsiders to z_min (score of the weakest top-k) instead of all top-k.
      incremental_topk=True
          Add inside-topk constraints incrementally (constraint generation).
      incremental_outside=True
          Add outsider-vs-topk constraints incrementally (ignored if outside_vs_mintop=True).

    Returns: (objective_value_L1, objective_value_rescaled_to_L2, alpha_unit_L2) or (None, None)
    """
    # ---- Model & solver knobs ----
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.Method = 2      # Barrier
    model.Params.Presolve = 2
    model.Params.Crossover = 1
    model.Params.Seed = 1
    model.Params.Threads = 1
    model.Params.NumericFocus = 2

    n, dims = len(points), len(points[0])

    # ---- Normalize top-k input into ordered tiers ----
    def _is_intlike(x):
        try:
            import numpy as _np
            return isinstance(x, (int,)) or isinstance(x, _np.integer)
        except Exception:
            return isinstance(x, int)

    if isinstance(topk_ranking, set):
        tiers = [list(int(i) for i in topk_ranking)]            # single tie tier
    elif hasattr(topk_ranking, "__iter__") and all(_is_intlike(x) for x in topk_ranking):
        tiers = [[int(i)] for i in topk_ranking]                # keep given order
    else:
        tiers = [[int(i) for i in tier] for tier in topk_ranking]

    ordered_topk = [i for tier in tiers for i in tier]
    topk_set = set(ordered_topk)
    outsiders = sorted(set(range(n)) - topk_set)

    tier_of = {}
    for ti, tier in enumerate(tiers):
        for idx in tier:
            tier_of[idx] = ti

    # ---- Variables ----
    alpha = model.addVars(range(dims), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")
    model.addConstr(gp.quicksum(alpha[d] for d in range(dims)) == 1)

    def score_expr(i):
        return gp.quicksum(alpha[d] * points[i][d] for d in range(dims))

    # lazy pairwise slacks (used inside-topk and pairwise outsiders if needed)
    error_vars = {}
    def get_err(i, j):
        if i > j: i, j = j, i
        if i not in error_vars:
            error_vars[i] = {}
        if j not in error_vars[i]:
            error_vars[i][j] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_{i}_{j}")
        return error_vars[i][j]

    obj = gp.LinExpr()

    # ---- z_min (optional) ----
    z_min = None
    error_out_to_z = {}
    if outside_vs_mintop:
        z_min = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="z_min")
        for tk in ordered_topk:
            model.addConstr(score_expr(tk) >= z_min)
        if strict:
            for o in outsiders:
                model.addConstr(score_expr(o) <= z_min)
        else:
            for o in outsiders:
                e = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_out_{o}")
                error_out_to_z[o] = e
                model.addConstr(e >= score_expr(o) - z_min + epsilon)
            obj += gp.quicksum(error_out_to_z.values())

    # ---- Seed: ties and cross-tier neighbours ----
    # ties: star per tier (NO epsilon)
    for tier in tiers:
        if len(tier) > 1:
            hub = tier[0]
            for j in tier[1:]:
                mi, mj = (hub, j) if hub < j else (j, hub)
                e = get_err(mi, mj)
                model.addConstr(e >= score_expr(hub) - score_expr(j))
                model.addConstr(e >= score_expr(j) - score_expr(hub))
                obj += e
 

    # ---- Outsiders when not using z_min ----
    if not outside_vs_mintop:
        if incremental_outside:
            if strict:
                outside_mode = "pairwise_cg"
            if outside_mode not in ("pairwise_cg", "aggregate_exact"):
                raise ValueError("outside_mode must be 'pairwise_cg' or 'aggregate_exact'")
            if outside_mode == "aggregate_exact" and not strict:
                k = len(ordered_topk)
                y = {}
                e_out_sum = {}
                for o in outsiders:
                    e_out_sum[o] = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"e_outsum_{o}")
                    so_plus = score_expr(o) + epsilon
                    rhs = k * so_plus
                    for t in ordered_topk:
                        y[(t, o)] = model.addVar(lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_{t}_{o}")
                        model.addConstr(y[(t, o)] <= score_expr(t))
                        model.addConstr(y[(t, o)] <= so_plus)
                        rhs -= y[(t, o)]
                    model.addConstr(e_out_sum[o] >= rhs)
                    obj += e_out_sum[o]
            # else: pure pairwise CG will be added in the loop
        else:
            # full add of all pairs
            for t in ordered_topk:
                for o in outsiders:
                    if strict:
                        model.addConstr(score_expr(t) - score_expr(o) >= 0.0)
                    else:
                        e = get_err(min(t, o), max(t, o))
                        model.addConstr(e >= score_expr(o) - score_expr(t) + epsilon)
                        obj += e

    

    # ---- Constraint generation loop  ----
    if incremental_topk:
        model.setObjective(obj, gp.GRB.MINIMIZE)
        margin_cross = 0.0 if strict else epsilon
        margin_out   = 0.0 if strict else epsilon

        added_cross = set()
        added_tie   = set()
        added_out   = set()

        # cross-tier: adjacent neighbours (+epsilon)
        for r in range(len(ordered_topk) - 1):
            i, j = ordered_topk[r], ordered_topk[r + 1]
            if tier_of[i] == tier_of[j]:
                continue
            mi, mj = (i, j) if i < j else (j, i)
            if strict:
                model.addConstr(score_expr(i) - score_expr(j) >= 0.0)
            else:
                e = get_err(mi, mj)
                model.addConstr(e >= score_expr(j) - score_expr(i) + epsilon)
                obj += e

        for i in error_vars:
            for j in error_vars[i]:
                if tier_of.get(i, -1) == tier_of.get(j, -1):
                    added_tie.add((i, j))
                else:
                    added_cross.add((i, j))

        for _ in range(max_rounds):
            model.optimize()
            if model.SolCount == 0:
                return (None, None)

            alpha_vals = [alpha[d].X for d in range(dims)]
            scores = [sum(alpha_vals[d] * points[i][d] for d in range(dims)) for i in range(n)]

            new_added = 0

            # (1) cross-tier: add violated earlier->later pairs
            for later_pos in range(len(ordered_topk)):
                t = ordered_topk[later_pos]
                for earlier_pos in range(later_pos):
                    c = ordered_topk[earlier_pos]
                    if tier_of[c] == tier_of[t]:
                        continue
                    if scores[t] - scores[c] + margin_cross > tol:
                        mi, mj = (c, t) if c < t else (t, c)
                        if (mi, mj) in added_cross:
                            continue
                        if strict:
                            model.addConstr(score_expr(c) - score_expr(t) >= 0.0)
                        else:
                            e = get_err(mi, mj)
                            model.addConstr(e >= score_expr(t) - score_expr(c) + margin_cross)
                            obj += e
                        added_cross.add((mi, mj))
                        new_added += 1

            # (2) ties within tier (no epsilon)
            for tier in tiers:
                if len(tier) <= 1:
                    continue
                for i, j in combinations(tier, 2):
                    mi, mj = (i, j) if i < j else (j, i)
                    if (mi, mj) in added_tie:
                        continue
                    if abs(scores[i] - scores[j]) > tol:
                        e = get_err(mi, mj)
                        model.addConstr(e >= score_expr(i) - score_expr(j))
                        model.addConstr(e >= score_expr(j) - score_expr(i))
                        obj += e
                        added_tie.add((mi, mj))
                        new_added += 1

            # (3) outsiders (only when not using z_min)
            if not outside_vs_mintop:
                if strict or outside_mode == "pairwise_cg":
                    for tk in ordered_topk:
                        s_tk = scores[tk]
                        for o in outsiders:
                            if scores[o] - s_tk + margin_out > tol:
                                mi, mj = (tk, o) if tk < o else (o, tk)
                                if (mi, mj) in added_out:
                                    continue
                                if strict:
                                    model.addConstr(score_expr(tk) - score_expr(o) >= 0.0)
                                else:
                                    e = get_err(mi, mj)
                                    model.addConstr(e >= score_expr(o) - score_expr(tk) + margin_out)
                                    obj += e
                                added_out.add((mi, mj))
                                new_added += 1
                # aggregate_exact case is already exact; nothing to add

            if new_added == 0:
                print(f"number of rounds {_} ")
                break
            if earlybreak:
               print(f"number of rounds {_} ")
               break

            model.setObjective(obj, gp.GRB.MINIMIZE)
    else:
        curr_proc = []
        for tkr in topk_ranking:
            # Add error computations for points which are ranked better than the current level of tkr
            for c in curr_proc:
                for t in tkr:
                    # No error if alpha.c - alpha.t >= epsilon
                    # Else : error is alpha.t - alpha.c + epsilon
                    # c has a lesser/better rank compared to t, so no error if c has a better score than t else add
                    e = get_err(min(c, t), max(c, t))
                    model.addConstr(e >= score_expr(t) - score_expr(c) + epsilon)
                    obj += e

            # Add items in topk level to current list
            for t in tkr:
                curr_proc.append(t)

            # If tkr is greater than one then all elements in same level should have same score, else penalise
            if len(tkr) > 1:
                # Do pair wise error computation
                for pairs in combinations(tkr, 2):
                    # Add error to the error function
                    i, j = pairs
                    mi, mj = min(i, j), max(i, j)
                    e = get_err(mi, mj)
                    model.addConstr(e >= score_expr(i) - score_expr(j))
                    model.addConstr(e >= score_expr(j) - score_expr(i))
                    obj += e


    # ---- Final solve & rescale (L1 -> L2) ----
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        return (None, None)

    alpha_vals = [alpha[d].X for d in range(dims)]
    l1 = sum(abs(v) for v in alpha_vals)
    assert abs(l1 - 1) <= 1e-4

    l2 = math.sqrt(sum(v*v for v in alpha_vals)) or 1.0
    alpha_unit = [v / l2 for v in alpha_vals]
    print("opt", model.ObjVal)
    return model.ObjVal, (model.ObjVal / l2), alpha_unit



# =======================
# Dataset loading helpers
# =======================
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_data_path(filename, explicit_path=None):
    """Find data files relative to this script, so CLI paths are optional."""
    candidates = []
    if explicit_path:
        p = Path(explicit_path).expanduser()
        candidates.extend([p, SCRIPT_DIR / p])
    name = Path(filename)
    candidates.extend([
        SCRIPT_DIR / "data" / name,
        SCRIPT_DIR / name,
        SCRIPT_DIR / "data" / "OnlineNewsPopularity" / name,
        SCRIPT_DIR / "OnlineNewsPopularity" / name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {filename!r}. Searched:\n  {searched}\n"
        f"Place the file under {SCRIPT_DIR / 'data'} or pass an explicit override."
    )


def _numeric_matrix(df, columns, dataset_name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name}: missing required columns: {missing}")
    out = df.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    out = out.loc[out.notna().all(axis=1)]
    if out.empty:
        raise ValueError(f"{dataset_name}: no complete numeric rows remain.")
    return out.astype(float)


def _maybe_scale(features, scale):
    if scale:
        features = pd.DataFrame(
            MinMaxScaler().fit_transform(features),
            columns=features.columns, index=features.index
        )
    return features.to_numpy(dtype=float)


def get_nba_dataset(csv_path=None, scale=False):
    """NBA: eight attributes, reference order descending MP*PER."""
    path = resolve_data_path("per.csv", csv_path)
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%", "MP*PER"]
    if not set(required).issubset(df.columns):
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"NBA CSV missing columns: {missing}")
    df["MP*PER"] = pd.to_numeric(df["MP*PER"], errors="coerce")
    df = df.dropna(subset=["MP*PER"]).sort_values("MP*PER", ascending=False, kind="mergesort")
    features = _numeric_matrix(
        df, ["PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%", "FT%"], "NBA"
    )
    return _maybe_scale(features, scale)


def get_jee_dataset(csv_path=None, scale=False):
    """JEE: stable descending mark sort, then five attributes."""
    path = resolve_data_path("jee.csv", csv_path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["category", "GENDER", "math", "phys", "chem", "mark"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"JEE CSV missing columns: {missing}")
    df["mark"] = pd.to_numeric(df["mark"], errors="coerce")
    df = df.dropna(subset=["mark"]).sort_values("mark", ascending=False, kind="mergesort")
    category_map = {"GEN": 0, "OBC-NCL": 1, "SC": 2, "ST": 3}
    gender_map = {"M": 0, "F": 1}
    df["category"] = df["category"].astype(str).str.strip().map(category_map)
    df["GENDER"] = df["GENDER"].astype(str).str.strip().map(gender_map)
    features = _numeric_matrix(
        df, ["category", "GENDER", "math", "phys", "chem"], "JEE"
    )
    return _maybe_scale(features, scale)


def get_reco_dataset(csv_path=None, scale=False, variant="mallows"):
    """Recommendation: seven visible features, reference order ascending rank."""
    filename = "reco_mallows.csv" if variant.lower() == "mallows" else "reco_thurstone.csv"
    path = resolve_data_path(filename, csv_path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = [
        "rank", "relevance_score", "popularity", "freshness", "predicted_ctr",
        "quality_score", "price", "diversity_tag"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Recommendation CSV missing columns: {missing}")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"]).sort_values("rank", ascending=True, kind="mergesort")
    cols = [
        "relevance_score", "popularity", "freshness", "predicted_ctr",
        "quality_score", "price", "diversity_tag"
    ]
    features = _numeric_matrix(df, cols, "Recommendation")
    return _maybe_scale(features, scale)


def get_csrankings_dataset(csv_path=None, scale=False):
    """CSRankings: 27 area features, reference order ascending rank."""
    path = resolve_data_path("csrankings_full.csv", csv_path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "rank" not in df.columns:
        raise ValueError("CSRankings CSV must contain 'rank'.")
    cols = [c for c in df.columns if c not in {"institution", "rank", "score"}]
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"]).sort_values("rank", ascending=True, kind="mergesort")
    features = _numeric_matrix(df, cols, "CSRankings")
    if len(cols) != 27:
        raise ValueError(f"Expected 27 CSRankings area features; found {len(cols)}.")
    return _maybe_scale(features, scale)


def get_online_news_dataset(csv_path=None, scale=False):
    """Online News: 58 predictive features, reference order descending shares."""
    path = resolve_data_path("OnlineNewsPopularity.csv", csv_path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "shares" not in df.columns:
        raise ValueError("Online News CSV must contain 'shares'.")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df = df.dropna(subset=["shares"]).sort_values("shares", ascending=False, kind="mergesort")
    cols = [c for c in df.columns if c not in {"url", "timedelta", "shares"}]
    features = _numeric_matrix(df, cols, "Online News")
    if len(cols) != 58:
        raise ValueError(f"Expected 58 Online News features; found {len(cols)}.")
    return _maybe_scale(features, scale)


def load_dataset(dataset, csv_override=None, scale=False):
    if dataset == "nba":
        return get_nba_dataset(csv_override, scale), "MP*PER descending"
    if dataset == "jee":
        return get_jee_dataset(csv_override, scale), "mark descending"
    if dataset == "reco_mallows":
        return get_reco_dataset(csv_override, scale=scale, variant="mallows"), "rank ascending"
    if dataset == "reco_thurstone":
        return get_reco_dataset(csv_override, scale=scale, variant="thurstone"), "rank ascending"
    if dataset == "csrankings":
        return get_csrankings_dataset(csv_override, scale), "rank ascending"
    if dataset == "online_news":
        return get_online_news_dataset(csv_override, scale), "shares descending"
    raise ValueError(f"Unknown dataset: {dataset}")


def find_hyperplane_through_points(points):
    """
    Given d points in d-dimensional space, find the hyperplane equation coefficients.

    The hyperplane equation is: w^T x + b = 0,
    where [w, b] is the solution of the system:
    [p1^T 1]
    [p2^T 1] * [w; b] = 0
    [ ...   ]
    [pd^T 1]

    Parameters:
    points: numpy array of shape (d, d) - d points in d-dimensional space

    Returns:
    w: numpy array of shape (d,) - normal vector to the hyperplane
    b: float - offset term
    """
    d = points.shape[1]
    # Construct matrix A by appending 1 to each point
    A = np.hstack((points, np.ones((d, 1))))

    # Find the null space (kernel) of A, i.e., vector [w; b] such that A @ [w; b] = 0
    # Use SVD for numerical stability
    U, S, Vt = np.linalg.svd(A)
    # The null space vector corresponds to the last row of Vt (smallest singular value)
    null_space_vector = Vt[-1, :]

    w = null_space_vector[:-1]
    b = null_space_vector[-1]


    if np.all(w<0):
      w *= -1
      b *= -1
    return w, b

def get_intersection_point(w, b):
  lambda_val = abs(-b)/sum(map(lambda x:x*x, w))
  point = lambda_val*w
  norm_point = point/np.linalg.norm(point)
  return norm_point

def farthest_point(w, b):

  # Create model
  model = gp.Model()
  dims = len(w)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  # Equation of the ball, all inclusive
  model.addConstr(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)) <= 1)

  # Equation of plane
  model.addConstr(gp.quicksum(w[i]*alpha[i] for i in range(dims)) + b == 0)

  model.setObjective(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)), gp.GRB.MINIMIZE)
  model.optimize()

  # Check if Optimization Successful!
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    # Step 1 : Obtain the alpha function

    # Check that alpha_vars l1 norm adds to 1
    l1_norm = sum(alpha_vals[i]* w[i] for i in range(dims)) + b
    assert abs(l1_norm) <= 0.0001

    # Get optimization score value
    optimal_value = model.ObjVal
    return math.sqrt(optimal_value), alpha_vals
  else:
    # Optimization score does not exist
    return None, None

def non_convex_opt(points, topk_ranking, strict=False, epsilon = 0.0001):

  # Create model
  model = gp.Model()

  n = len(points)
  dims = len(points[0])

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for pairs in combinations(list(range(n)), 2):
    i, j = min(pairs), max(pairs)

    if i not in error_vars:
      error_vars[i] = dict()

    # Do not allow negative errors
    error_vars[i][j] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"error_{i}_{j}", lb=0.0)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  # Add the constraint that sum of alpha squares is 1
  model.addConstr(gp.quicksum(alpha[i]*alpha[i] for i in range(dims)) == 1)

  curr_proc = []
  for tkr in topk_ranking:
    # Add error computations for points which are ranked better than the current level of tkr
    for c in curr_proc:
      for t in tkr:
        # No error if alpha.c - alpha.t >= epsilon
        # Else : error is alpha.t - alpha.c + epsilon
        # c has a lesser/better rank compared to t, so no error if c has a better score than t else add
        model.addConstr(error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)


    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      # Do pair wise error computation
      for pairs in combinations(tkr, 2):
        # Add error to the error function
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))

  # For all items not in topk need to add error constraint or a satisfying constraint that topk are better
  # for tk in topk_ranking:
  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        # Consider these as violations if tk is not better than ntk by atleast epsilon
        model.addConstr(error_vars[min(tk, ntk), max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)
  model.optimize()

  # Check if Optimization Successful!
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    # Step 1 : Obtain the alpha function

    # Check that alpha_vars l1 norm adds to 1
    l2_norm = sum(alpha_vals[i]*alpha_vals[i] for i in range(dims))
    assert abs(l2_norm-1) <= 0.0001

    # Get optimization score value
    optimal_value = model.ObjVal
    return optimal_value, alpha_vals
  else:
    # Optimization score does not exist
    return None, None

def dlppp(points, topk_ranking, strict=False, epsilon = 0.0001):
  n = len(points)
  dims = len(points[0])

  # Generate empty parameters
  min_error = math.inf
  min_alpha = []

  # Create model
  model = gp.Model()

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for pairs in combinations(list(range(n)), 2):
    i, j = min(pairs), max(pairs)

    if i not in error_vars:
      error_vars[i] = dict()

    # Do not allow negative errors
    error_vars[i][j] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"error_{i}_{j}", lb=0.0)

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  curr_proc = []
  for tkr in topk_ranking:
    # Add error computations for points which are ranked better than the current level of tkr
    for c in curr_proc:
      for t in tkr:
        # No error if alpha.c - alpha.t >= epsilon
        # Else : error is alpha.t - alpha.c + epsilon
        # c has a lesser/better rank compared to t, so no error if c has a better score than t else add
        model.addConstr(error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)


    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      # Do pair wise error computation
      for pairs in combinations(tkr, 2):
        # Add error to the error function
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))

  # For all items not in topk need to add error constraint or a satisfying constraint that topk are better
  # for tk in topk_ranking:
  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        # Consider these as violations if tk is not better than ntk by atleast epsilon
        try:
          model.addConstr(error_vars[min(tk, ntk)][max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
        except:
          import pdb
          pdb.set_trace()
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)

  plane_constr = None

  for dim in range(dims):
    if plane_constr is not None:
      model.remove(plane_constr)

    # Add the constraints for a plane such that it goes through d points.
    # Repalce the dim th point with (1/sqrt(d), ..., 1/sqrt(d))

    # dim was dims-1 then the equa
    # w1 + w2 + ... + w(d−1) − (d − 1 − sqrt(d))wd = 1

    # Add the constraint that sum of alphas is 1
    plane_constr = model.addConstr(gp.quicksum(alpha[i] if not(i == dim) else -(dims-1-math.sqrt(dims))*alpha[i] for i in range(dims)) == 1)

    model.optimize()

    # Check if Optimization Successful!
    if model.SolCount > 0:
      alpha_vals = [alpha[i].X for i in range(dims)]

      # Step 1 : Obtain the alpha function

      # Check that equation
      # w1 + w2 + ... + w(d−1) − (d − 1 − sqrt(d))wd = 1
      # is satisfied
      l1_norm = sum(alpha_vals[i] if not(i == dim) else -(dims-1-math.sqrt(dims))*alpha_vals[i] for i in range(dims))
      assert abs(l1_norm-1) <= 0.0001

      # Step 2. Rescale the function to move from the l1 norm to the l2 norm
      l2_norm = math.sqrt(sum(math.pow(alpha_vals[i], 2) for i in range(dims)))
      alpha_vals = [alpha_vals[i]/l2_norm for i in range(dims)]

      # Get optimization score value
      optimal_value = model.ObjVal/l2_norm
      if optimal_value < min_error:
        min_error = optimal_value
        min_alpha = alpha_vals
    else:
      # Optimization score does not exist
      # Something is wrong!
      return None, None
  # Get maximum error
  return min_error, min_alpha

def ilp_inv(points, topk_ranking, strict=False, epsilon = 0.0001):

  n = len(points)
  dims = len(points[0])

  # Create model
  model = gp.Model()

  alpha = model.addVars(list(range(dims)), lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, name="alpha")

  # Equation of the plane on which alpha lies in
  model.addConstr(gp.quicksum(alpha[i] for i in range(dims)) == 1)

  # As a large constant, using dims for now. Ideally, need to use
  # the l_2 norm of all points and choose the largest.
  large_const = 2*dims

  # Get items in topk
  topk = set()
  for tkr in topk_ranking:
    for item in tkr:
      topk.add(item)

  # Get not in top k
  not_in_topk = set(range(n)) - topk

  error_vars = dict()
  for ti in topk:
    for ntj in range(n):
      if ti == ntj:
        continue
      i, j = min(ti, ntj), max(ti, ntj)

      if i not in error_vars:
        error_vars[i] = dict()

      # Do not allow negative errors
      error_vars[i][j] = model.addVar(vtype=gp.GRB.BINARY, name=f"error_{i}_{j}", lb=0)

  curr_proc = []
  for tkr in topk_ranking:
    # Add error computations for points which are ranked better than the current level of tkr
    for c in curr_proc:
      for t in tkr:
        # No error if alpha.c - alpha.t >= epsilon
        # Else : error is alpha.t - alpha.c + epsilon
        # c has a lesser/better rank compared to t, so no error if c has a better score than t else add
        model.addConstr(large_const*error_vars[min(c, t)][max(c, t)] >= gp.quicksum(alpha[_i]*(points[t][_i] - points[c][_i]) for _i in range(dims)) + epsilon)
    # Add items in topk level to current list
    for t in tkr:
      curr_proc.append(t)

    # If tkr is greater than one then all elements in same level should have same score, else penalise
    if len(tkr) > 1:
      # Do pair wise error computation
      for pairs in combinations(tkr, 2):
        # Add error to the error function
        i, j = pairs
        mi, mj = min(i, j), max(i, j)
        # As ti and tj are in same level, they will not be penalised only if
        # both ti and tj are ranked/scored equally.
        model.addConstr(large_const*error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[i][_i] - points[j][_i]) for _i in range(dims)))
        model.addConstr(large_const*error_vars[mi][mj] >= gp.quicksum(alpha[_i]*(points[j][_i] - points[i][_i]) for _i in range(dims)))


  # For all items not in topk need to add error constraint or a satisfying constraint that topk are better
  # for tk in topk_ranking:
  for tk in itertools.chain.from_iterable(topk_ranking):
    for ntk in not_in_topk:
      if strict:
        model.addConstr(gp.quicksum(alpha[_i]*(points[tk][_i] - points[ntk][_i]) for _i in range(dims)) >= 0)
      else:
        # Consider these as violations if tk is not better than ntk by atleast epsilon
        model.addConstr(large_const*error_vars[min(tk, ntk), max(tk, ntk)] >= gp.quicksum(alpha[_i]*(points[ntk][_i] - points[tk][_i]) for _i in range(dims)) + epsilon)
  model.setObjective(gp.quicksum(error_vars[i][j] for i in range(n) for j in range(i + 1 ,n)), gp.GRB.MINIMIZE)
  model.optimize()

  # Check if Optimization Successful!
  if model.SolCount > 0:
    alpha_vals = [alpha[i].X for i in range(dims)]

    # Step 1 : Obtain the alpha function

    # Check that alpha_vars l1 norm adds to 1
    l1_norm = sum(alpha_vals[i] for i in range(dims))
    assert abs(l1_norm-1) <= 0.0001

    # Get optimization score value
    optimal_value = model.ObjVal
    return optimal_value, alpha_vals
  else:
    # Optimization score does not exist
    return None, None

def partial_shuffle(seq, tk, seed=None):
    """
    Permute only a fraction of positions in `seq` in-place.
    Guarantees all selected positions change (via 1-step rotation).
    """
    rng = random.Random(seed)
    n = len(seq)

    # choose k distinct positions
    k = int(1.5 * tk)
    idxs = rng.sample(range(min(k, n)), tk)
    idxs.sort()  # order doesn’t matter; sort for reproducibility
    # rotate the chosen values by 1 so no element stays in place
    vals = [seq[i] for i in idxs]
    vals_rot = vals[-1:] + vals[:-1]
    for i, v in zip(idxs, vals_rot):
        seq[i] = v
    return seq

# =======================
# Linear-regression baseline helpers
# =======================

def make_targets_from_order(n, ordered_topk, outsiders, high=1.0, low=0.0):
    """
    Assign a decreasing target in [low, high] over ordered_topk.
    Outsiders get 'low'.
    """
    y = np.full(n, low, dtype=float)
    k = len(ordered_topk)
    if k > 1:
        for r, i in enumerate(ordered_topk):
            y[i] = high - (high - low) * (r / (k - 1))
    elif k == 1:
        y[ordered_topk[0]] = high
    return y

def fit_linear_regression_simplex(points, y):
    """
    Constrained least squares on the simplex:
        min ||X alpha - y||^2
        s.t. alpha >= 0, sum(alpha) = 1
    Returns:
        alpha_unit (L2-normalized for diagnostics), alpha_simplex (raw)
    """
    X = np.asarray(points, dtype=float)  # n x d
    n, d = X.shape
    y = np.asarray(y, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y
    Q = 2.0 * XtX
    c = -2.0 * Xty

    m = gp.Model()
    m.Params.OutputFlag = 0

    alpha = m.addMVar(d, lb=0.0, name="alpha")  # non-negative; no need for ub
    m.addConstr(alpha.sum() == 1.0)

    # Quadratic objective: 0.5 * alpha^T Q alpha + c^T alpha
    obj = gp.QuadExpr()
    for i in range(d):
        # diagonal and off-diagonal
        if Q[i, i] != 0.0:
            obj += 0.5 * Q[i, i] * alpha[i] * alpha[i]
        for j in range(i + 1, d):
            if Q[i, j] != 0.0:
                obj += Q[i, j] * alpha[i] * alpha[j]  # 0.5 on (i,j) + 0.5 on (j,i) -> add once without 0.5
        if c[i] != 0.0:
            obj += c[i] * alpha[i]

    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return None, None

    alpha_simplex = np.array(alpha.X, dtype=float)
    l2 = np.linalg.norm(alpha_simplex) or 1.0
    alpha_unit = (alpha_simplex / l2).tolist()
    return alpha_unit, alpha_simplex.tolist()

# =======================
# Linear-regression baseline (weighted variant)
# =======================
def fit_linear_regression_simplex_weighted(points, y, w=None):
    """
    Weighted constrained least squares on the simplex.
    Computes X.T @ W @ X without ever constructing the n-by-n matrix W.
    """
    X = np.asarray(points, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    if y.shape != (n,):
        raise ValueError(f"Expected y shape ({n},), got {y.shape}.")
    weights = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be a finite, nonnegative vector of length n.")

    Xw = X * weights[:, None]
    Q = 2.0 * (X.T @ Xw)
    c = -2.0 * (X.T @ (weights * y))

    m = gp.Model("LR_weighted_least_squares")
    m.Params.OutputFlag = 0
    alpha = m.addMVar(d, lb=0.0, name="alpha")
    m.addConstr(alpha.sum() == 1.0)
    obj = gp.QuadExpr()
    for i in range(d):
        if Q[i, i] != 0.0:
            obj += 0.5 * Q[i, i] * alpha[i] * alpha[i]
        for j in range(i + 1, d):
            if Q[i, j] != 0.0:
                obj += Q[i, j] * alpha[i] * alpha[j]
        if c[i] != 0.0:
            obj += c[i] * alpha[i]
    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        return None, None
    alpha_simplex = np.asarray(alpha.X, dtype=float)
    norm = np.linalg.norm(alpha_simplex) or 1.0
    return (alpha_simplex / norm).tolist(), alpha_simplex.tolist()

# =======================
# Main
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the constrained linear-regression ranking-explanation baseline."
    )
    parser.add_argument("--dataset", choices=[
        "nba", "jee", "reco_mallows", "reco_thurstone", "csrankings", "online_news"
    ], default="reco_mallows")
    parser.add_argument("--items", type=int, default=1000,
                        help="Number of highest-ranked rows to use.")
    parser.add_argument("--tk", type=int, default=100,
                        help="Number of top-ranked rows.")
    parser.add_argument("--dims", type=int, default=None,
                        help="Optional number of leading features to use.")
    parser.add_argument("--scale", type=int, choices=[0, 1], default=0,
                        help="MinMax-scale features; default 0 preserves previous behavior.")
    parser.add_argument("--nba_csv", default=None)
    parser.add_argument("--jee_csv", default=None)
    parser.add_argument("--reco_csv", default=None)
    parser.add_argument("--csrankings_csv", default=None)
    parser.add_argument("--online_news_csv", default=None)
    parser.add_argument("--lr_baseline", choices=["none", "unweighted", "weighted"],
                        default="weighted")
    parser.add_argument("--lr_high", type=float, default=1.0)
    parser.add_argument("--lr_low", type=float, default=0.0)
    parser.add_argument("--lr_topk_weight", type=float, default=1.0)
    parser.add_argument("--lr_outsider_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    overrides = {
        "nba": args.nba_csv,
        "jee": args.jee_csv,
        "reco_mallows": args.reco_csv,
        "reco_thurstone": args.reco_csv,
        "csrankings": args.csrankings_csv,
        "online_news": args.online_news_csv,
    }
    data, reference_order = load_dataset(
        args.dataset, csv_override=overrides[args.dataset], scale=bool(args.scale)
    )

    if args.items <= 0 or args.items > len(data):
        parser.error(f"--items must be in [1, {len(data)}] for {args.dataset}.")
    if args.tk <= 0 or args.tk > args.items:
        parser.error(f"--tk must be in [1, {args.items}].")
    if args.dims is not None:
        if args.dims <= 0 or args.dims > data.shape[1]:
            parser.error(f"--dims must be in [1, {data.shape[1]}].")
        data = data[:, :args.dims]

    data = np.asarray(data[:args.items], dtype=float)
    n, d = data.shape
    topk_ranking = [[i] for i in range(args.tk)]

    print("=" * 60)
    print("LINEAR REGRESSION RANKING-EXPLANATION BASELINE")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Reference order: {reference_order}")
    print(f"n={n}, k={args.tk}, d={d}")
    print(f"Feature scaling: {'MinMax' if args.scale else 'None'}")
    print(f"LR variant: {args.lr_baseline}")

    if args.lr_baseline == "none":
        raise SystemExit("LR baseline disabled.")

    ordered_topk = list(range(args.tk))
    y = make_targets_from_order(
        n, ordered_topk, list(range(args.tk, n)),
        high=args.lr_high, low=args.lr_low
    )

    start_time = time.time()
    if args.lr_baseline == "unweighted":
        alpha_unit, alpha_simplex = fit_linear_regression_simplex(data, y)
    else:
        weights = np.full(n, args.lr_outsider_weight, dtype=float)
        weights[:args.tk] = args.lr_topk_weight
        alpha_unit, alpha_simplex = fit_linear_regression_simplex_weighted(
            data, y, w=weights
        )
    elapsed = time.time() - start_time

    if alpha_unit is None:
        raise SystemExit("Gurobi did not return a feasible solution.")

    diagnostics = lppp_diagnostics_counts(
        data.tolist(), topk_ranking, alpha_unit, strict=False, epsilon=1e-4
    )
    print(f"Runtime: {elapsed:.6f} seconds")
    print("Simplex utility weights:", alpha_simplex)
    print("L2-normalized utility weights:", alpha_unit)
    print("Evaluation diagnostics:", diagnostics)

