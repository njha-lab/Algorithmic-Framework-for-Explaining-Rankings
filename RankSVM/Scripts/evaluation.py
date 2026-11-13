"""
evaluation.py

Evaluation module for all ranking algorithms.

This file implements the error measures introduced in

Finding Linear Explanations for a Given Ranking

The evaluator is completely independent of RankSVM.

Any algorithm that produces a linear weight vector can
use this evaluator.

Author: Nidhi Jha
"""

import csv
import numpy as np
import pandas as pd


# ============================================================
# Weight vector
# ============================================================

def load_weight_vector(model_file):
    """
    Reads the learned weight vector from the SVMrank model.

    Parameters
    ----------
    model_file : str

    Returns
    -------
    numpy.ndarray
    """

    weights = {}

    with open(model_file, "r") as f:

        lines = f.readlines()

    #
    # Support vectors start after the header.
    #

    for line in lines[11:]:

        tokens = line.strip().split()

        if len(tokens) == 0:
            continue

        alpha = float(tokens[0])

        for token in tokens[1:]:

            if token.startswith("#"):
                break

            if ":" not in token:
                continue

            feature, value = token.split(":")

            feature = int(feature)

            value = float(value)

            weights[feature] = (
                weights.get(feature, 0.0)
                + alpha * value
            )

    dimension = max(weights.keys())

    w = np.zeros(dimension)

    for i in range(1, dimension + 1):

        w[i - 1] = weights.get(i, 0.0)

    return w


# ============================================================
# Normalize
# ============================================================

def normalize_weight_vector(w):
    """
    Returns a normalized weight vector.

    The paper assumes

            ||W|| = 1
    """

    norm = np.linalg.norm(w)

    if norm == 0:

        return w

    return w / norm


# ============================================================
# Scores
# ============================================================

def compute_scores(
        df,
        feature_columns,
        weight_vector,
):
    """
    Computes

        score = W^T X

    using the NORMALIZED weight vector.
    """

    X = df[feature_columns].to_numpy(dtype=float)

    return X @ weight_vector


# ============================================================
# Ground-truth ranking
# ============================================================

def compute_ground_truth_rank(
        df,
        ranking_column,
        higher_is_better,
):
    """
    Computes π.

    Competition ranking is used.

    Example

    scores

    10
    9
    9
    7

    ranks

    1
    2
    2
    4
    """

    values = df[ranking_column].to_numpy()

    order = np.argsort(values)

    if higher_is_better:

        order = order[::-1]

    rank = np.zeros(len(values), dtype=int)

    current_rank = 1

    rank[order[0]] = current_rank

    for i in range(1, len(values)):

        previous = order[i - 1]

        current = order[i]

        if values[current] == values[previous]:

            rank[current] = current_rank

        else:

            current_rank = i + 1

            rank[current] = current_rank

    return rank


# ============================================================
# Predicted ranking
# ============================================================

def compute_predicted_rank(scores):
    """
    Computes ρ.

    Higher score = better rank.

    Competition ranking.

    Predicted ties receive
    the same rank.
    """

    order = np.argsort(-scores)

    rank = np.zeros(len(scores), dtype=int)

    current_rank = 1

    rank[order[0]] = current_rank

    for i in range(1, len(scores)):

        previous = order[i - 1]

        current = order[i]

        if np.isclose(

                scores[current],

                scores[previous],

                atol=1e-12,

        ):

            rank[current] = current_rank

        else:

            current_rank = i + 1

            rank[current] = current_rank

    return rank


# ============================================================
# Top-k tuples
# ============================================================

def topk_indices(
        ground_rank,
        k,
):
    """
    Returns the indices of the FIRST k tuples
    in the ground-truth ranking.

    This follows the definition of D_pi(k)
    used in the paper.

    Ground-truth ties are NOT expanded.
    """

    order = np.argsort(ground_rank)

    return order[:k]


# ============================================================
# Pretty printing
# ============================================================

def print_weight_vector(
        feature_columns,
        weight_vector,
):
    """
    Displays the normalized weight vector.
    """

    print()

    print("=" * 60)

    print("Normalized Weight Vector")

    print("=" * 60)

    for feature, weight in zip(

            feature_columns,

            weight_vector,

    ):

        print(

            f"{feature:20s}"

            f"{weight:12.6f}"

        )

    print()

    print(

        "||W|| =",

        np.linalg.norm(weight_vector),

    )

    print("=" * 60)
# ============================================================
# Pairwise Penalty (PP)
# ============================================================

def compute_PP(
        ground_rank,
        scores,
        topk,
):
    """
    Computes Pairwise Penalty (PP).

    Definition (paper):

    PP =
        Σ ( score(s) - score(r) )

    where

        r ∈ Top-k
        s ∈ Whole dataset

        π(r) < π(s)

        score(s) > score(r)

    Ground-truth ties are ignored.

    Predicted ties are NOT violations.
    """

    top = topk_indices(
        ground_rank,
        topk,
    )

    n = len(scores)

    pp = 0.0

    for r in top:

        for s in range(n):

            #
            # Ignore itself
            #

            if r == s:
                continue

            #
            # Ignore ground-truth ties
            #

            if ground_rank[r] == ground_rank[s]:
                continue

            #
            # r must be better
            #

            if ground_rank[r] > ground_rank[s]:
                continue

            #
            # violation
            #

            if scores[s] > scores[r]:

                pp += (
                    scores[s]
                    -
                    scores[r]
                )

    return pp


# ============================================================
# Inversion Count (INV)
# ============================================================

def compute_INV(
        ground_rank,
        scores,
        topk,
):
    """
    Computes INV.

    Counts every violated pair.

    r ∈ Top-k

    s ∈ Whole ranking

    Ground-truth ties ignored.

    Predicted ties ignored.
    """

    top = topk_indices(
        ground_rank,
        topk,
    )

    n = len(scores)

    inside_inv = 0
    outsider_inv = 0

    for r in top:

        for s in range(n):

            if r == s:
                continue

            if ground_rank[r] == ground_rank[s]:
                continue

            if ground_rank[r] > ground_rank[s]:
                continue

            if scores[s] > scores[r]:
                if s in set(top):
                    inside_inv += 1
                else:
                    outsider_inv += 1

    return inside_inv + outsider_inv


# ============================================================
# Minimum Position Error (MPE)
# ============================================================

def compute_MPE(
        ground_rank,
        predicted_rank,
        topk,
):
    """
    Computes

        Σ |ρ(r)-π(r)|

    over Top-k.

    Competition ranking
    is assumed.
    """

    top = topk_indices(
        ground_rank,
        topk,
    )

    mpe = 0

    for r in top:

        mpe += abs(

            predicted_rank[r]

            -

            ground_rank[r]

        )

    return mpe


# ============================================================
# Largest Individual Position Error (LIPE)
# ============================================================

def compute_LIPE(
        ground_rank,
        predicted_rank,
        topk,
):
    """
    Computes

        max |ρ(r)-π(r)|

    over Top-k.
    """

    top = topk_indices(
        ground_rank,
        topk,
    )

    lipe = 0

    for r in top:

        error = abs(

            predicted_rank[r]

            -

            ground_rank[r]

        )

        if error > lipe:

            lipe = error

    return lipe
# ============================================================
# Print evaluation
# ============================================================

def print_results(results):
    """
    Pretty print the evaluation measures.
    """

    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"PP   : {results['PP']:.6f}")
    print(f"INV  : {results['INV']}")
    print(f"MPE  : {results['MPE']}")
    print(f"LIPE : {results['LIPE']}")

    print("=" * 60)


# ============================================================
# Save evaluation
# ============================================================

def save_results(
        filename,
        algorithm,
        dataset,
        topk,
        results,
):
    """
    Saves results to CSV.

    If the file already exists,
    append a new row.
    """

    import os

    exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "Algorithm",
                "Dataset",
                "TopK",
                "PP",
                "INV",
                "MPE",
                "LIPE",
            ])

        writer.writerow([
            algorithm,
            dataset,
            topk,
            results["PP"],
            results["INV"],
            results["MPE"],
            results["LIPE"],
        ])


# ============================================================
# Evaluate
# ============================================================

def evaluate(
        df,
        feature_columns,
        ranking_column,
        higher_is_better,
        weight_vector,
        topk,
):
    """
    Main evaluation routine.
    """

    #
    # Normalize
    #

    weight_vector = normalize_weight_vector(
        weight_vector
    )

    #
    # Print vector
    #

    print_weight_vector(
        feature_columns,
        weight_vector,
    )

    #
    # Compute scores
    #

    scores = compute_scores(
        df,
        feature_columns,
        weight_vector,
    )

    #
    # Compute rankings
    #

    ground_rank = compute_ground_truth_rank(
        df,
        ranking_column,
        higher_is_better,
    )

    predicted_rank = compute_predicted_rank(
        scores,
    )

    #
    # Compute measures
    #

    results = {

        "PP":

            compute_PP(
                ground_rank,
                scores,
                topk,
            ),

        "INV":

            compute_INV(
                ground_rank,
                scores,
                topk,
            ),

        "MPE":

            compute_MPE(
                ground_rank,
                predicted_rank,
                topk,
            ),

        "LIPE":

            compute_LIPE(
                ground_rank,
                predicted_rank,
                topk,
            ),
    }

    print_results(results)

    return results