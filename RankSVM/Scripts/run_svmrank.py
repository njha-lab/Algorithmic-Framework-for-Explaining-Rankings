"""
run_svmrank.py

Main driver for RankSVM experiments.

Pipeline

CSV
    ↓
csv_to_svmrank.py
    ↓
train.dat
    ↓
svm_rank_learn.exe
    ↓
model.dat
    ↓
svm_rank_classify.exe
    ↓
predictions.txt
"""

import argparse
import subprocess
import sys
import time

import pandas as pd

from evaluation import (
    load_weight_vector,
    evaluate,
    save_results,
)

from config import (
    SCRIPT_DIR,
    DATASET_DIR,
    RESULT_DIR,
    TRAIN_FILE,
    MODEL_FILE,
    PREDICTION_FILE,
    LEARN_EXE,
    CLASSIFY_EXE,
)


# ==========================================================
# Utility
# ==========================================================

def run_command(command):

    print("\n=================================================")
    print("Running command")
    print("=================================================\n")

    print(" ".join(str(x) for x in command))
    print()

    result = subprocess.run(command)

    if result.returncode != 0:

        print("\nERROR: command failed.\n")

        sys.exit(1)


# ==========================================================
# Main
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Complete RankSVM pipeline."
    )

    # ---------------------------------------------
    # Dataset
    # ---------------------------------------------

    parser.add_argument(
        "--dataset",
        required=True,
    )

    # ---------------------------------------------
    # Feature columns
    # ---------------------------------------------

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
    )

    # ---------------------------------------------
    # Ranking column
    # ---------------------------------------------

    parser.add_argument(
        "--ranking",
        required=True,
    )

    # ---------------------------------------------
    # Mode
    # ---------------------------------------------

    parser.add_argument(
        "--mode",
        default="full",
        choices=[
            "full",
            "topk",
            "existing",
        ],
    )

    # ---------------------------------------------
    # Top-k
    # ---------------------------------------------

    parser.add_argument(
        "--k",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Use only the first n tuples.",
    )

    # ---------------------------------------------
    # Ranking direction
    # ---------------------------------------------

    parser.add_argument(
        "--higher_is_better",
        action="store_true",
    )

    # ---------------------------------------------
    # RankSVM C parameter
    # ---------------------------------------------

    parser.add_argument(
        "--C",
        type=float,
        default=20.0,
    )

    args = parser.parse_args()

    # In the JEE dataset, a larger mark always means a better rank.
    # Enforce this convention even if --higher_is_better was omitted.
    if args.ranking.strip().lower() == "mark":
        args.higher_is_better = True

    # ======================================================
    # Step 1
    # CSV -> train.dat
    # ======================================================

    converter = SCRIPT_DIR / "csv_to_svmrank.py"

    command = [

        sys.executable,

        str(converter),

        "--dataset",

        args.dataset,

        "--features",

    ]

    command.extend(args.features)

    command.extend([

        "--ranking",

        args.ranking,

        "--mode",

        args.mode,

    ])

    if args.k is not None:

        command.extend([

            "--k",

            str(args.k),

        ])

    if args.n is not None:

        command.extend([

            "--n",

            str(args.n),

        ])
    

    if args.higher_is_better:

        command.append(

            "--higher_is_better"

        )

    run_command(command)

    # ======================================================
    # Step 2
    # Train RankSVM
    # ======================================================

    _t_train = time.time()

    run_command([

        str(LEARN_EXE),

        "-c",

        str(args.C),

        str(TRAIN_FILE),

        str(MODEL_FILE),

    ])

    _train_seconds = time.time() - _t_train

    # ======================================================
    # Step 3
    # Predict
    # ======================================================

    run_command([

        str(CLASSIFY_EXE),

        str(TRAIN_FILE),

        str(MODEL_FILE),

        str(PREDICTION_FILE),

    ])
    # ======================================================
    # Evaluation
    # ======================================================

    print("\nEvaluating learned weight vector...")

    #
    # Read dataset again
    #

    dataset_path = DATASET_DIR / args.dataset

    df = pd.read_csv(dataset_path, low_memory=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Mirror csv_to_svmrank.py exactly. For JEE, sort by mark BEFORE
    # truncating to n, so training and evaluation use the same top-n rows.
    if args.ranking.strip().lower() == "mark":
        if args.ranking not in df.columns:
            raise ValueError("JEE ranking column 'mark' is missing from the CSV.")
        df[args.ranking] = pd.to_numeric(df[args.ranking], errors="coerce")
        df = df.dropna(subset=[args.ranking]).sort_values(
            args.ranking,
            ascending=not args.higher_is_better,
            kind="mergesort",
        ).reset_index(drop=True)

    if args.n is not None:
        if args.n < 1:
            raise ValueError("--n must be positive.")
        df = df.head(args.n).copy()

    for feature in args.features:
        if feature not in df.columns:
            raise ValueError(f"Feature column {feature!r} is missing.")
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df[args.ranking] = pd.to_numeric(df[args.ranking], errors="coerce")
    df = df.dropna(subset=list(args.features) + [args.ranking]).copy()

    if args.mode in {"full", "topk"}:
        df = df.sort_values(
            args.ranking,
            ascending=not args.higher_is_better,
            kind="mergesort",
        ).reset_index(drop=True)
    # mode='existing' preserves the CSV row order, just like the converter.

    effective_n = len(df)
    effective_k = min(args.k, effective_n) if args.mode == "topk" and args.k is not None else effective_n
    if args.mode == "topk" and args.k is None:
        raise ValueError("--k is required when --mode topk.")
    if effective_k < 1:
        raise ValueError("No usable rows remain after applying the training filters.")

    #
    # Load learned vector
    #

    weight_vector = load_weight_vector(
        MODEL_FILE
    )

    #
    # Evaluate
    #

    results = evaluate(

        df=df,

        feature_columns=args.features,

        ranking_column=args.ranking,

        higher_is_better=args.higher_is_better,

        weight_vector=weight_vector,

        topk=effective_k,

    )

    #
    # Save results
    #

    save_results(

        filename=RESULT_DIR / "results.csv",

        algorithm="RankSVM",

        dataset=args.dataset,

        topk=effective_k,

        results=results,

    )

    # one parseable line for collect_logs.py (the metrics otherwise only go to the shared Results/results.csv)
    print("SVMrank summary", {"n": int(effective_n), "k": int(effective_k), "PP": float(results["PP"]), "INV": int(results["INV"]),
                              "MPE": int(results["MPE"]), "LIPE": int(results["LIPE"]), "train_seconds": float(_train_seconds)})

    # ======================================================
    # Finished
    # ======================================================

    print("\n========================================")

    print("Pipeline completed successfully.")

    print("========================================\n")

    print("Train file:")

    print(TRAIN_FILE)

    print()

    print("Model file:")

    print(MODEL_FILE)

    print()

    print("Prediction file:")

    print(PREDICTION_FILE)


if __name__ == "__main__":

    main()