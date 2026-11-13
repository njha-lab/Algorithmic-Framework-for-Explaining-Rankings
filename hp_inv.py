

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import kendalltau
import argparse
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import time



@dataclass
class QuadtreeNode:
    points: List[int]
    ranking: Optional[List[int]] = None
    children: Optional[List['QuadtreeNode']] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    def is_leaf(self): return not self.children


class ShiftedQuadtree:
    def __init__(self, data: np.ndarray, max_leaf_size: int = 1):
        self.data = data
        self.n, self.d = data.shape
        self.max_leaf_size = max_leaf_size

    def build(self):
        all_indices = list(range(self.n))
        minb, maxb = np.min(self.data, axis=0), np.max(self.data, axis=0)
        return self._rec(all_indices, minb, maxb)

    def _rec(self, idxs, minb, maxb):
        # Stop when the leaf is small enough.
        if len(idxs) <= self.max_leaf_size:
            return QuadtreeNode(points=list(idxs), bounds=(minb, maxb))

        mid = (minb + maxb) / 2.0
        # Assign each point to exactly one child. Equality goes to the upper
        # half, avoiding the overlap caused by inclusive bounds in all children.
        groups = {}
        for i in idxs:
            p = self.data[i]
            mask = 0
            for dim in range(self.d):
                if p[dim] >= mid[dim]:
                    mask |= (1 << dim)
            groups.setdefault(mask, []).append(i)

        # If no strict partition is possible (e.g. all points coincide, or
        # floating-point ranges collapse), retain a leaf instead of recursing.
        nonempty = [g for g in groups.values() if g]
        if len(nonempty) <= 1 or any(len(g) == len(idxs) for g in nonempty):
            return QuadtreeNode(points=list(idxs), bounds=(minb, maxb))

        kids = []
        for mask, subidx in sorted(groups.items()):
            if not subidx:
                continue
            submin, submax = minb.copy(), maxb.copy()
            for dim in range(self.d):
                if (mask >> dim) & 1:
                    submin[dim] = mid[dim]
                else:
                    submax[dim] = mid[dim]
            # Defensive progress check: every recursive child must be a
            # strict subset of its parent.
            if len(subidx) >= len(idxs):
                continue
            kids.append(self._rec(subidx, submin, submax))

        if not kids:
            return QuadtreeNode(points=list(idxs), bounds=(minb, maxb))
        return QuadtreeNode(points=list(idxs), children=kids, bounds=(minb, maxb))

# =========================
#  Linear Ranking Optimizer
# =========================

class LinearRankingOptimizer:
    def __init__(self, data: np.ndarray, ranking: np.ndarray, top_k: Optional[int] = None,
                 fast_merge: bool = False):
        self.data = np.asarray(data)
        self.ranking = np.asarray(ranking)
        self.n, self.d = self.data.shape
        assert self.ranking.ndim == 1 and len(self.ranking) == self.n
        self.top_k = top_k if top_k else self.n
        self.fast_merge = fast_merge

        # Reuse one Gurobi environment
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()

    # ---------- LP feasibility (time-limited) ----------
    def check_feasibility_lp(self, order, eps=1e-3):
        try:
            X, d = self.data, self.data.shape[1]
            m = gp.Model("feas", env=self.env)
            m.Params.OutputFlag = 0
            m.Params.TimeLimit = 0.05
            m.Params.Presolve, m.Params.Threads = 1, 1
            w = m.addVars(d, lb=0)
            for i in range(len(order) - 1):
                i1, i2 = order[i], order[i+1]
                m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d)) >= eps)
            m.addConstr(gp.quicksum(w[k] for k in range(d)) == 1)
            m.optimize()
            return m.status == GRB.OPTIMAL
        except Exception as e:
            print("LP err:", e)
            return False

    # ---------- Kendall distance (vectorized) ----------
    def kendall_tau_distance(self, r1, r2):
        common = sorted(set(r1) & set(r2))
        if len(common) <= 1: return 0.0
        p1 = {v:i for i,v in enumerate(r1) if v in common}
        p2 = {v:i for i,v in enumerate(r2) if v in common}
        a, b = [p1[v] for v in common], [p2[v] for v in common]
        tau, _ = kendalltau(a, b)
        return 0.5*(1-(tau or 0))

    def compute_ranking_error(self, r, ref):
        order = [i for i in np.argsort(ref) if i in set(r)]
        return self.kendall_tau_distance(r, order)

    # ---------- Greedy merge (uses --fast_merge policy) ----------
    def greedy_merge(self, L1, L2, ref):
        res = L1.copy()
        for it in L2:
            if it in res: continue
            best_r, best_err = None, float('inf')
            
            for pos in range(len(res)+1):
                cand = res[:pos]+[it]+res[pos:]
                err = self.compute_ranking_error(cand, ref)
                if self.fast_merge:  # skip LPs during merge
                    if err < best_err: best_err, best_r = err, cand
                else:
                    if err <= best_err and self.check_feasibility_lp(cand):
                        best_err, best_r = err, cand
            res = best_r if best_r else res+[it]
        return res, self.compute_ranking_error(res, ref)

    def greedy_merge_window(self, base_order, to_insert, ref, window_size: int = 25):
        """Insert items by only probing a small window around their ref rank."""
        res = base_order.copy()
        pos_ref = {i: r for r, i in enumerate(np.argsort(ref))}  # ref rank positions
        for it in to_insert:
            if it in res: continue
            # Target position based on reference ranking
            t = pos_ref[it]
            # Clamp window to current list bounds
            left = max(0, min(len(res), t) - window_size)
            right = min(len(res), max(0, min(len(res), t) + window_size))
            best_r, best_err = None, float('inf')
            # Probe only within [left, right]
            for pos in range(left, right+1):
                cand = res[:pos]+[it]+res[pos:]
                err = self.compute_ranking_error(cand, ref)
                if self.fast_merge:
                    if err < best_err: best_err, best_r = err, cand
                else:
                    if err <= best_err and self.check_feasibility_lp(cand):
                        best_err, best_r = err, cand
            res = best_r if best_r else res+[it]
        return res

    def merge_rankings(self, L1, L2, ref):
        m1,e1=self.greedy_merge(L1,L2,ref)
        m2,e2=self.greedy_merge(L2,L1,ref)
        return m1 if e1<=e2 else m2

    def bottom_up_merge(self,node,ref):
        if node.is_leaf(): return sorted(node.points,key=lambda x:ref[x])
        ch=[self.bottom_up_merge(c,ref) for c in node.children]
        res=ch[0]
        for c in ch[1:]: res=self.merge_rankings(res,c,ref)
        node.ranking=res
        return res

    
    def extract_weights(self, ranking):
        X,d=self.data,self.data.shape[1]
        m=gp.Model("weights",env=self.env)
        m.Params.OutputFlag=0
        m.Params.Presolve,m.Params.Threads,m.Params.TimeLimit=1,1,5
        w=m.addVars(d,lb=0)
        eps=1e-3
        free = getattr(self, "lp_margin", "strict") == "free"
        if not free:   # original behaviour: every adjacent pair must win by >= eps, otherwise the LP is infeasible
            for i in range(len(ranking)-1):
                i1,i2=ranking[i],ranking[i+1]
                m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d))>=eps)
        m.addConstr(gp.quicksum(w[k] for k in range(d))==1)
        s=m.addVar(lb=-GRB.INFINITY if free else 0)   # free: the best achievable margin may be negative
        for i in range(len(ranking)-1):
            i1,i2=ranking[i],ranking[i+1]
            m.addConstr(gp.quicksum(w[k]*(X[i1,k]-X[i2,k]) for k in range(d))>=s)
        m.setObjective(s,GRB.MAXIMIZE)
        m.optimize()
        if m.status==GRB.OPTIMAL: return np.array([w[k].X for k in range(d)])
        print("warn: fallback weights"); return np.ones(d)/d

   
    def optimize(self, tree_scope: str = "all", leaf_size: int = 16,
                 insert_mode: str = "baseline", window_size: int = 25):
        """
        tree_scope: "all" (build on n) or "topk" (build on top-k, then insert rest)
        insert_mode (for tree_scope='topk'):
            - 'baseline': greedy insert (slowest)
            - 'window':   probe only ±window_size positions per item
            - 'score':    extract weights from top-k order, score all items, sort by score
        """
        if tree_scope not in ("all","topk"):
            raise ValueError("tree_scope must be 'all' or 'topk'")
        if insert_mode not in ("baseline","window","score"):
            raise ValueError("insert_mode must be 'baseline','window','score'")

        if tree_scope == "all":
            print("Building quadtree on ALL points...")
            tree=ShiftedQuadtree(self.data,max_leaf_size=leaf_size)
            root=tree.build()
            print("Merging...")
            final=self.bottom_up_merge(root,self.ranking)

        else:
            # --- Build/merge on top-k only (local view), then insert remaining globally ---
            print("Building quadtree on TOP-K points...")
            topk_idx = np.argsort(self.ranking)[:self.top_k]              # global ids of top-k
            X_top = self.data[topk_idx]                                    # (TK, d)
            ref_top = self.ranking[topk_idx]                               # local ref ranks
            tree=ShiftedQuadtree(X_top,max_leaf_size=leaf_size)
            root=tree.build()
            print("Merging (top-k only)...")
            local_order = self.bottom_up_merge(root, ref_top)              # local ids 0..TK-1
            topk_global_order = [int(topk_idx[i]) for i in local_order]    # map to global

            remaining = [i for i in range(self.n) if i not in set(topk_global_order)]
            print(f"Remaining to insert: {len(remaining)} (mode={insert_mode})")

            if insert_mode == "baseline":
                final, _ = self.greedy_merge(topk_global_order, remaining, self.ranking)

            elif insert_mode == "window":
                final = self.greedy_merge_window(topk_global_order, remaining,
                                                 self.ranking, window_size=window_size)

            else:  # insert_mode == 'score'
                # 1) Extract weights from the top-k order only
                print("Extracting provisional weights from top-k...")
                w_top = self.extract_weights(topk_global_order)
                # 2) Score everyone once
                scores = self.data @ w_top
                # 3) Final = sort all ids by score descending (fastest)
                final = list(np.argsort(-scores))

        if insert_mode == "score" and tree_scope == "topk" and getattr(self, "final_weights", "extract") == "wtop":
            print("Using the weights that sorted the items as final weights (no second LP).")
            w = w_top
        else:
            print("Extracting final weights...")
            w=self.extract_weights(final)
        err=self.kendall_tau_distance(final,list(np.argsort(self.ranking)[:self.top_k]))
        print(f"Final Kendall distance (vs top-k ref) = {err:.4f}")
        l2=math.sqrt(sum(v*v for v in w)) or 1
        return np.array([v/l2 for v in w]),final


def _normalize_tiers(t):
    if isinstance(t,set):return[sorted(list(t))]
    if isinstance(t,list):
        if not t:return[]
        if all(isinstance(x,int) for x in t):return[[i]for i in t]
        if all(isinstance(x,list)for x in t):return[ list(x) for x in t]
    raise ValueError

def lppp_diagnostics_counts(points,topk,alpha,strict=False,epsilon=1e-4,tol=1e-12):
    n,d=len(points),len(points[0])
    scores=[sum(alpha[k]*points[i][k]for k in range(d))for i in range(n)]
    tiers=_normalize_tiers(topk)
    top=[i for ti in tiers for i in ti]; topS=set(top)
    outs=[i for i in range(n) if i not in topS]
    cnt,inv=0.0,0;seen=[]
    for ti in tiers:
        for t in ti:
            for c in seen:
                gap=scores[t]-scores[c]
                if gap>0 and gap+epsilon>tol:inv+=1;cnt+=gap
        seen+=ti
    pair,outsAny=0,set()
    for o in outs:
        so=scores[o];v=False
        for t in top:
            gap=so-scores[t]
            if gap>0 and gap+epsilon>tol:pair+=1;cnt+=gap;v=True
        if v:outsAny.add(o)
    rank=sorted(range(n),key=lambda i:scores[i],reverse=True)
    posS={i:r for r,i in enumerate(rank)}
    posG={i:r for r,i in enumerate(top)}
    absd=[abs(posS[i]-posG[i])for i in top]
    return {"pp":float(cnt),"topk_inversions_count":inv,
            "outside_pair_violations":pair,
            "full_inv":inv+pair,
            "outsiders_with_any_violation":len(outsAny),
            "max_abs_rank_diff":max(absd)if absd else 0,
            "sum_abs_rank_diff":int(sum(absd)) if absd else 0}


def get_nba_dataset():
    df=pd.read_csv('./data/per.csv',skiprows=1,header=0).dropna()
    df=df[df.columns[1:9]]
    sc=MinMaxScaler()
    dfS=pd.DataFrame(sc.fit_transform(df),columns=df.columns)
    return dfS.values

def get_reco_dataset(csv_path: str, scale=True):
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


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="nba",
                   choices=["nba","jee","reco","reco_thurstone","reco_mallows","csrankings","online_news"])
    p.add_argument("--items",type=int,default=1000)
    p.add_argument("--dims",type=int,default=None,help="Use the first DIMS features for dimensionality experiments.")
    p.add_argument("--reco_csv", type=str, default="./data/reco.csv",
                    help="Path to recommendation synthetic CSV when --dataset reco")
    p.add_argument("--jee_csv", type=str, default="./data/jee.csv",
                   help="Path to JEE CSV when --dataset jee")
    p.add_argument("--jee_scale", type=int, default=1,
                   help="If 1, MinMax-scale the JEE marks (same default as main.py)")
    p.add_argument("--reco_thurstone_csv", type=str, default="./data/reco_thurstone.csv",
                   help="Path to the Thurstone synthetic CSV when --dataset reco_thurstone")
    p.add_argument("--reco_mallows_csv", type=str, default="./data/reco_mallows.csv",
                   help="Path to the Mallows synthetic CSV when --dataset reco_mallows")
    p.add_argument("--csrankings_csv", type=str, default="./data/csrankings_full.csv",
                   help="Path to the CSRankings CSV when --dataset csrankings")
    p.add_argument("--news_csv", type=str, default="./data/OnlineNewsPopularity/OnlineNewsPopularity.csv",
                   help="Path to OnlineNewsPopularity.csv when --dataset online_news")
    p.add_argument("--news_feature_set", choices=["all"], default="all",
                   help="Online News feature set (all = 58 predictive features)")
    p.add_argument("--tk",type=int,default=5)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--verbose",type=int,default=0)
    p.add_argument("--fast_merge",type=int,default=1,
                   help="If 1, skip LP checks during insertion")
    p.add_argument("--tree_scope",choices=["all","topk"],default="topk",
                   help="Build quadtree on all n points or only top-k, then insert the rest")
    p.add_argument("--leaf_size",type=int,default=16,
                   help="Quadtree leaf size (larger => fewer nodes)")
    p.add_argument("--insert_mode",choices=["baseline","window","score"],default="score",
                   help="When tree_scope=topk: how to insert remaining points")
    p.add_argument("--lp_margin",choices=["strict","free"],default="strict",
                   help="strict: original LP (margin >= 1e-3 on every pair, else fallback to uniform weights); free: margin may be negative")
    p.add_argument("--final_weights",choices=["extract","wtop"],default="extract",
                   help="extract: re-solve an LP on the whole score-sorted ranking (original); wtop: keep the weights that produced the scores")
    p.add_argument("--window_size",type=int,default=25,
                   help="Window size for insert_mode=window")
    args=p.parse_args()
    random.seed(args.seed)

    if args.dataset=="nba":
        pts=get_nba_dataset()[:args.items]
        rank=np.arange(args.items)                         # simple rank vector: 0 best, 1 next, ...
        topk=[[i] for i in range(args.tk)]                # for diagnostics

        print("insert mode", args.insert_mode)
        start=time.time()
        if args.dims is not None:
            if args.dims < 1 or args.dims > np.asarray(pts).shape[1]:
                raise ValueError(f"--dims must be between 1 and {np.asarray(pts).shape[1]}.")
            pts = np.asarray(pts, dtype=float)[:, :args.dims]
        opt=LinearRankingOptimizer(pts,rank,args.tk,fast_merge=bool(args.fast_merge))
        opt.lp_margin=args.lp_margin; opt.final_weights=args.final_weights
        alpha,final=opt.optimize(tree_scope=args.tree_scope,
                                 leaf_size=args.leaf_size,
                                 insert_mode=args.insert_mode,
                                 window_size=args.window_size)
        print(f"Runtime={time.time()-start:.2f}s")
        summ=lppp_diagnostics_counts(pts.tolist(),topk,alpha)
        print("summary",summ)
    
    if args.dataset == "reco":
        data = get_reco_dataset(args.reco_csv, scale=False)
        if len(data) < args.items:
          raise ValueError(f"Recommendation dataset has only {len(data)} rows.")
        pts = data[:args.items]
        rank=np.arange(args.items)                         # simple rank vector: 0 best, 1 next, ...
        topk=[[i] for i in range(args.tk)]                # for diagnostics

        print("insert mode", args.insert_mode)
        start=time.time()
        if args.dims is not None:
            if args.dims < 1 or args.dims > np.asarray(pts).shape[1]:
                raise ValueError(f"--dims must be between 1 and {np.asarray(pts).shape[1]}.")
            pts = np.asarray(pts, dtype=float)[:, :args.dims]
        opt=LinearRankingOptimizer(pts,rank,args.tk,fast_merge=bool(args.fast_merge))
        opt.lp_margin=args.lp_margin; opt.final_weights=args.final_weights
        alpha,final=opt.optimize(tree_scope=args.tree_scope,
                                 leaf_size=args.leaf_size,
                                 insert_mode=args.insert_mode,
                                 window_size=args.window_size)
        print(f"Runtime={time.time()-start:.2f}s")
        summ=lppp_diagnostics_counts(pts,topk,alpha)
        print("summary",summ)

    if args.dataset in ("jee", "reco_thurstone", "reco_mallows", "csrankings", "online_news"):
        if args.dataset == "jee":
            data = get_jee_dataset(args.jee_csv, scale=bool(args.jee_scale))
        elif args.dataset == "reco_thurstone":
            data = get_reco_dataset(args.reco_thurstone_csv, scale=False)
        elif args.dataset == "reco_mallows":
            data = get_reco_dataset(args.reco_mallows_csv, scale=False)
        elif args.dataset == "csrankings":
            data = get_csrankings_dataset(args.csrankings_csv, scale=False)
        else:
            data = get_online_news_dataset(args.news_csv, scale=True, feature_set=args.news_feature_set)
        if len(data) < args.items:
            raise ValueError(f"{args.dataset} dataset has only {len(data)} usable rows; --items is {args.items}.")
        if args.tk > args.items:
            raise ValueError(f"--tk ({args.tk}) cannot exceed --items ({args.items}).")
        pts = data[:args.items]
        rank=np.arange(args.items)                         # simple rank vector: 0 best, 1 next, ...
        topk=[[i] for i in range(args.tk)]                # for diagnostics

        print("insert mode", args.insert_mode)
        start=time.time()
        if args.dims is not None:
            if args.dims < 1 or args.dims > np.asarray(pts).shape[1]:
                raise ValueError(f"--dims must be between 1 and {np.asarray(pts).shape[1]}.")
            pts = np.asarray(pts, dtype=float)[:, :args.dims]
        opt=LinearRankingOptimizer(pts,rank,args.tk,fast_merge=bool(args.fast_merge))
        opt.lp_margin=args.lp_margin; opt.final_weights=args.final_weights
        alpha,final=opt.optimize(tree_scope=args.tree_scope,
                                 leaf_size=args.leaf_size,
                                 insert_mode=args.insert_mode,
                                 window_size=args.window_size)
        print(f"Runtime={time.time()-start:.2f}s")
        summ=lppp_diagnostics_counts(pts,topk,alpha)
        print("summary",summ)
