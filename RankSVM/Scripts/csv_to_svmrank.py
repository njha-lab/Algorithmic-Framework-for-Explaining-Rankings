"""
Standalone CSV -> RankSVM converter.

Example

python Scripts/csv_to_svmrank.py ^
    --dataset jee.csv ^
    --features math phys chem ^
    --ranking mark ^
    --higher_is_better

Example (Top-k)

python Scripts/csv_to_svmrank.py ^
    --dataset syn.csv ^
    --features relevance_score popularity freshness predicted_ctr quality_score price diversity_tag ^
    --ranking rank ^
    --mode topk ^
    --k 100
"""

import argparse
import pandas as pd

from converter import (
    remove_unnamed_columns,
    validate_features,
    assign_relevance_full,
    assign_relevance_topk,
    assign_relevance_existing,
    convert_to_svmrank,
    preview,
)

from config import (
    DATASET_DIR,
    TRAIN_FILE,
)


def main():

    parser = argparse.ArgumentParser(
        description="Convert CSV dataset to RankSVM format."
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    parser.add_argument(
        "--dataset",
        required=True,
        help="CSV file inside Datasets folder",
    )

    # --------------------------------------------------
    # Feature columns
    # --------------------------------------------------

    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature column names",
    )

    # --------------------------------------------------
    # Ranking column
    # --------------------------------------------------

    parser.add_argument(
        "--ranking",
        required=True,
        help="Ranking column (or relevance column)",
    )

    # --------------------------------------------------
    # Mode
    # --------------------------------------------------

    parser.add_argument(
        "--mode",
        choices=["full", "topk", "existing"],
        default="full",
    )

    # --------------------------------------------------
    # Top-k
    # --------------------------------------------------

    parser.add_argument(
        "--k",
        type=int,
        default=None,
    )

    # --------------------------------------------------
    # Number of tuples
    # --------------------------------------------------

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Use only the first n tuples.",
    )


    # --------------------------------------------------
    # Ranking direction
    # --------------------------------------------------

    parser.add_argument(
        "--higher_is_better",
        action="store_true",
        help="Higher ranking values indicate better objects.",
    )

    args = parser.parse_args()

    # In the JEE dataset, a larger mark always means a better rank.
    # Enforce this convention even if --higher_is_better was omitted.
    if args.ranking.strip().lower() == "mark":
        args.higher_is_better = True

    # --------------------------------------------------
    # Read dataset
    # --------------------------------------------------

    dataset_path = DATASET_DIR / args.dataset

    print("\nReading dataset...")

    print(dataset_path)

    df = pd.read_csv(
        dataset_path,
        low_memory=False,
    )
    
    # --------------------------------------------------
    # JEE reference order: descending marks, stable on ties.
    # Sort BEFORE --n so that --n selects the highest-mark tuples,
    # not merely the first rows in the unsorted source CSV.
    # --------------------------------------------------

    if args.ranking.strip().lower() == "mark":
        if args.ranking not in df.columns:
            raise ValueError("JEE ranking column 'mark' is missing from the CSV.")
        df[args.ranking] = pd.to_numeric(df[args.ranking], errors="coerce")
        df = df.dropna(subset=[args.ranking]).sort_values(
            args.ranking,
            ascending=not args.higher_is_better,
            kind="mergesort",
        ).reset_index(drop=True)
        print("\nJEE rows sorted by mark (descending; stable for ties).")

    # --------------------------------------------------
    # Restrict number of tuples AFTER the JEE ranking sort
    # --------------------------------------------------

    if args.n is not None:
        if args.n < 1:
            raise ValueError("--n must be positive.")
        print(f"\nUsing first {args.n} tuples")
        df = df.head(args.n).copy()

    # --------------------------------------------------
    # Remove unnamed columns
    # --------------------------------------------------

    df = remove_unnamed_columns(df)

    # --------------------------------------------------
    # Validate features
    # --------------------------------------------------

    df = validate_features(
        df,
        args.features,
    )
    df[args.ranking] = pd.to_numeric(df[args.ranking], errors="coerce")
    df = df.dropna(subset=[args.ranking]).copy()

    # --------------------------------------------------
    # Assign relevance
    # --------------------------------------------------

    if args.mode == "full":

        print("\nUsing FULL ranking")

        df = assign_relevance_full(
            df,
            ranking_column=args.ranking,
            higher_is_better=args.higher_is_better,
        )

    elif args.mode == "topk":

        if args.k is None:

            raise ValueError(
                "Please specify --k in topk mode."
            )

        print(f"\nUsing TOP-{args.k}")

        df = assign_relevance_topk(
            df,
            ranking_column=args.ranking,
            higher_is_better=args.higher_is_better,
            k=args.k,
        )

    else:

        print("\nUsing EXISTING relevance labels")

        df = assign_relevance_existing(
            df,
            relevance_column=args.ranking,
        )

    # --------------------------------------------------
    # Preview
    # --------------------------------------------------

    preview(df)

    # --------------------------------------------------
    # Convert
    # --------------------------------------------------

    convert_to_svmrank(
        df=df,
        output_file=TRAIN_FILE,
        feature_columns=args.features,
    )

    print("\nDone!")

    print(f"\nRankSVM file created at:\n{TRAIN_FILE}")


if __name__ == "__main__":
    main()