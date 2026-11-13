# Prereqs
- Python 3.9+, Gurobi (licensed)
- Data: `./data/per.csv` (NBA), `./data/jee.csv` (JEE), `./data/csrankings_full.csv`, `./data/OnlineNewsPopularity/OnlineNewsPopularity.csv`.
- Data: the synthetic `reco` datasets are generated (see below).
# Install
```
pip install numpy pandas scikit-learn scipy gurobipy
```

# To generate synthetic datasets
- three ranking models over the same items/features: hidden bonuses (`reco`), Thurstone noise (`reco_thurstone`), Mallows (`reco_mallows`).
```
python gen_data.py --lambda_1 0.1 --lambda_2 0.1 --ranking_model hidden_score --output ./data/reco.csv
python gen_data.py --ranking_model thurstone --output ./data/reco_thurstone.csv
python gen_data.py --ranking_model mallows --output ./data/reco_mallows.csv
```
Other flags: `--n` (1000000), `--seed` (42), `--thurstone_noise` (0.2), `--mallows_theta` (1.0).

# LP-PP & dP-PP

**LP-PP (sample run (nba/reco/jee dataset)):**
```bash
python main.py --dataset nba --items 1000 --tk 100 --lppp 1 --dppp 0 --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 1
python main.py --dataset reco --items 1000 --tk 100 --lppp 1 --dppp 0 --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 1
python main.py --dataset jee --items 20000 --tk 500 --lppp 1 --dppp 0 --outside_vs_mintop 1 --incremental_topk 1 --earlybreak 1
```
**dP-PP (sample run):**
```bash
python main.py --dataset nba --items 1000 --tk 100 --lppp 0 --dppp 1
```

## Core Flags (defaults in **bold**)
- `--dataset {nba,jee,reco,reco_thurstone,reco_mallows,csrankings,online_news}` (**nba**)
- `--items INT` (**1000**), `--tk INT` (**100**), `--dims INT` (use only the first d features; default: all)
- **LP-PP only:** `--outside_vs_mintop {0,1}` (**1**), `--incremental_topk {0,1}` (**1**), `--earlybreak {0,1}` (**1**)
  (the paper's `<OVM, INC, EB>`; e.g. the unoptimized LP-PP is `0 0 0`)
- Runner select: `--lppp {0,1}` (**1**), `--dppp {0,1}` (**0**)
- Misc: `--seed INT` (**10**), `--verbose {0,1}` (**0**)

## Outputs
- Learned `alpha` (unit-L2), objective value, runtime, one line per LP-PP iteration.
- Diagnostics (PP, inversions, outsider violations, rank diffs).
- Logs in `app.log`.

# GPP

```
python gpp.py --dataset jee --items 10000 --tk 5 --dims 5 --eps 0.5
```
## Core Flags:
`--eps` (**0.5**): approximation parameter; `--dims` (**3**): number of attributes (set it explicitly).

# Exact OPT*

```
python exact_opt.py --tk 6 --mode topk
python exact_opt.py --dataset nba --items 100 --tk 25 --mode topk --time_limit 3500
```
The first command is the running example of the paper. `--dataset {running_example,nba,jee,reco,online_news}` (**running_example**), `--time_limit` (**60**).
Prints a certified interval for OPT*.

# LS-INV

```
python ls_inv.py --dataset jee --items 1000 --tk 100 --jee_scale 1
```
## Core Flags:
In addition to --items and --tk, we have:
`--samples` (**1000**), which represents random samples for initial weights. Note: `--jee_scale` defaults to 0 here (1 in the other scripts); pass `--jee_scale 1`.

# HP-INV

```
python hp_inv.py --dataset reco_thurstone --items 1000 --tk 5 --tree_scope topk --insert_mode score --fast_merge 1 --final_weights wtop --lp_margin free
```
Full-tree merge (no insertion phase):

```
python hp_inv.py --dataset reco_thurstone --items 1000 --tk 5 --tree_scope all --lp_margin free
```
`--lp_margin free` is always feasible; the default `strict` falls back to uniform weights when its LP is infeasible.

## Core Flags:
In addition to --items and --tk, we have:
- `--tree_scope {all,topk}` (**topk**): build the tree over all items or only the top-k
- `--insert_mode {baseline,window,score}` (**score**): how the tuples of one child are inserted into the other during a merge
- `--fast_merge {0,1}` (**1**), `--leaf_size INT` (**16**), `--window_size INT` (**25**, for `window`)
- `--final_weights {extract,wtop}` (**extract**): how the final weight vector is obtained
- `--lp_margin {strict,free}` (**strict**)

# SVMrank (baselines/RankSVM)

Joachims' SVMrank with wrapper scripts (`Scripts/run_svmrank.py`, `csv_to_svmrank.py`, `evaluation.py`, `config.py`), modified for our setting:
`--mode topk` trains on the pairs inside the top-k and between the top-k and the rest, `--n` uses the first n items, and the metrics (PP, INV, MPE, LIPE) are computed like in `main.py`.
```
cd baselines/RankSVM && bash build_linux.sh
cd ../.. && python prepare_ranksvm_data.py --jee data/jee.csv --reco data/reco_thurstone.csv --reco_top 10000 --out baselines/RankSVM/Datasets/
cd baselines/RankSVM
python Scripts/run_svmrank.py --dataset jee_fileorder_ranksvm.csv --features category GENDER math phys chem --ranking rank --mode topk --k 1000 --n 10000 --C 20
python Scripts/run_svmrank.py --dataset reco_thurstone_ranksvm.csv --features relevance_score popularity freshness predicted_ctr quality_score price diversity_tag --ranking rank --mode topk --k 100 --n 1000 --C 20
```
- The features are min-max scaled (SVMrank does not converge on the raw ones), so the printed PP is in scaled units; `python Scripts/svmrank_raw_pp.py --raw_csv ../../data/reco_thurstone.csv` gives PP in raw units.

