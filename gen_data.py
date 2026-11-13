#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def _sample_mallows_permutation(n, theta, rng):
    """Sample a permutation from a Mallows model using Kendall distance.

    The central ranking is represented by [0, 1, ..., n-1]. The sampled
    permutation pi has probability proportional to exp(-theta * K(pi)),
    where K(pi) is its number of inversions relative to that central ranking.

    Uses an inversion-vector sampler and a Fenwick tree, taking O(n log n)
    time and O(n) memory. theta=0 gives a uniform random permutation; larger
    theta concentrates more strongly around the central ranking.
    """
    if n <= 1:
        return np.arange(n, dtype=int)
    if theta < 0:
        raise ValueError("mallows_theta must be non-negative")

    # Inversion-vector entry v_i can be any integer in {0, ..., i-1}.
    # Its probability is proportional to exp(-theta * v_i).
    inversion_vector = np.zeros(n, dtype=np.int64)
    if theta == 0 or theta < 1e-8:
        # At theta=0 the distribution is exactly uniform. For extremely
        # small theta, uniform sampling is a numerically stable approximation.
        for i in range(1, n + 1):
            inversion_vector[i - 1] = rng.integers(0, i)
    else:
        u = rng.random(n)
        i_values = np.arange(1, n + 1, dtype=float)
        # Inverse CDF of the truncated geometric distribution:
        # P(v_i=v) proportional to exp(-theta*v), v=0,...,i-1.
        truncation = -np.expm1(-theta * i_values)  # 1 - exp(-theta*i)
        sampled = -np.log1p(-u * truncation) / theta
        inversion_vector = np.floor(sampled).astype(np.int64)
        inversion_vector = np.minimum(inversion_vector, i_values.astype(np.int64) - 1)
        inversion_vector = np.maximum(inversion_vector, 0)

    # Fenwick tree initially represents n empty slots, each with count 1.
    # tree[i] = i & -i is the Fenwick representation of an all-ones array.
    tree = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        tree[i] = i & -i

    def select_kth_empty(k):
        """Return the 1-based index of the k-th currently empty slot."""
        index = 0
        bit = 1 << (n.bit_length() - 1)
        while bit:
            nxt = index + bit
            if nxt <= n and tree[nxt] < k:
                index = nxt
                k -= tree[nxt]
            bit >>= 1
        return index + 1

    permutation = np.empty(n, dtype=int)
    # Place labels from largest to smallest. For label i, exactly v_i smaller
    # labels must appear after it, so it occupies the (i - v_i)-th empty slot.
    for i in range(n, 0, -1):
        slot = select_kth_empty(i - int(inversion_vector[i - 1]))
        permutation[slot - 1] = i - 1  # zero-based central-rank label

        # Mark this slot as occupied in the Fenwick tree.
        j = slot
        while j <= n:
            tree[j] -= 1
            j += j & -j

    return permutation


def generate_dataset(
    n=1000,
    lambda_1=0.7,
    boost_strength=1.2,
    lambda_2=0.5,
    diversity_penalty_strength=1.0,
    n_categories=8,
    seed=42,
    ranking_model="hidden_score",
    thurstone_noise=0.2,
    mallows_theta=1.0,
):
    rng = np.random.default_rng(seed)

    # -----------------------------
    # Visible features
    # -----------------------------
    relevance_score = rng.beta(5, 2, size=n)
    popularity = rng.lognormal(mean=1.8, sigma=0.8, size=n)
    freshness = rng.beta(2, 5, size=n)
    predicted_ctr = np.clip(
        0.45 * relevance_score + 0.25 * minmax(popularity) + 0.30 * freshness,
        0, 1
    )
    quality_score = np.clip(
        0.50 * relevance_score + 0.30 * predicted_ctr + 0.20 * rng.random(n),
        0, 1
    )
    price = rng.lognormal(mean=3.2, sigma=0.6, size=n)

    # category-like numeric tag
    diversity_tag = rng.integers(0, n_categories, size=n)

    # -----------------------------
    # Visible score WX
    # -----------------------------
    popularity_norm = minmax(popularity)
    price_norm = minmax(price)

    X = np.column_stack([
        relevance_score,
        popularity_norm,
        freshness,
        predicted_ctr,
        quality_score,
        1.0 - price_norm,
        diversity_tag / max(1, n_categories - 1),
    ])

    W = np.array([0.28, 0.14, 0.12, 0.18, 0.16, 0.08, 0.04], dtype=float)
    W = W / W.sum()

    visible_score = X @ W
    visible_score = minmax(visible_score)

    # -----------------------------
    # Hidden boost signal
    # promotes sponsored/new items
    # make it continuous, not only binary
    # -----------------------------
    sponsored = rng.binomial(1, 0.15, size=n)
    sponsor_strength = sponsored * rng.uniform(0.6, 1.0, size=n)

    boost_signal = (
        0.55 * freshness
        + 0.25 * predicted_ctr
        + 0.20 * sponsor_strength
    )
    boost_signal = minmax(boost_signal)

    # -----------------------------
    # Hidden diversity penalty signal
    # penalize repetitive categories
    # make some categories much more common
    # -----------------------------
    category_popularity = rng.dirichlet(alpha=np.ones(n_categories) * 0.5)
    category_weight = category_popularity[diversity_tag]
    category_weight = minmax(category_weight)

    # more popular categories get more penalty
    # and highly popular items in such categories get penalized more
    diversity_penalty_signal = category_weight * (0.6 + 0.4 * popularity_norm)
    diversity_penalty_signal = minmax(diversity_penalty_signal)

    # -----------------------------
    # Final hidden score
    # all terms are on comparable scale
    # -----------------------------
    hidden_score = (
        visible_score
        + lambda_1 * boost_strength * boost_signal
        - lambda_2 * diversity_penalty_strength * diversity_penalty_signal
    )

    # -----------------------------
    # Ranking-generation model
    # -----------------------------
    # hidden_score: preserve the original procedure exactly: rank directly
    # by visible_score + hidden boost - hidden diversity penalty.
    # thurstone: add Gaussian random utility noise to visible_score, then sort.
    # mallows: sample a permutation around the ranking induced by visible_score.
    if ranking_model == "hidden_score":
        # Keep the original argsort/tie behavior for backward compatibility.
        order = np.argsort(-hidden_score)
    elif ranking_model == "thurstone":
        if thurstone_noise < 0:
            raise ValueError("thurstone_noise must be non-negative")
        score_std = float(np.std(visible_score))
        noise = rng.normal(
            loc=0.0,
            scale=thurstone_noise * score_std,
            size=n,
        )
        ranking_score = visible_score + noise
        # Stable sort: exact ties keep the original item order.
        order = np.argsort(-ranking_score, kind="stable")
    elif ranking_model == "mallows":
        if mallows_theta < 0:
            raise ValueError("mallows_theta must be non-negative")
        # Stable sorting defines the central order from visible_score;
        # exact ties retain original item order.
        central_order = np.argsort(-visible_score, kind="stable")
        permutation = _sample_mallows_permutation(n, mallows_theta, rng)
        order = central_order[permutation]
    else:
        raise ValueError(
            "ranking_model must be one of: 'hidden_score', 'thurstone', 'mallows'"
        )

    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    df = pd.DataFrame({
        "item_id": np.arange(n),
        "relevance_score": relevance_score,
        "popularity": popularity,
        "freshness": freshness,
        "predicted_ctr": predicted_ctr,
        "quality_score": quality_score,
        "price": price,
        "diversity_tag": diversity_tag.astype(float),
        "visible_score": visible_score,
        "boost_signal": boost_signal,
        "diversity_penalty_signal": diversity_penalty_signal,
        "hidden_score": hidden_score,
        "rank": rank,
        "is_sponsored": sponsored,
        "ranking_model": ranking_model,
    })

    return df.sort_values("rank").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--lambda_1", type=float, default=1.7)
    parser.add_argument("--boost", type=float, default=1.2)
    parser.add_argument("--lambda_2", type=float, default=0.5)
    parser.add_argument("--diversity_penalty", type=float, default=1.0)
    parser.add_argument("--n_categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ranking_model",
        choices=["hidden_score", "thurstone", "mallows"],
        default="hidden_score",
        help=(
            "Ranking mechanism: hidden_score preserves the original method; "
            "thurstone adds Gaussian noise to visible_score; mallows samples "
            "a ranking around the visible-score ranking."
        ),
    )
    parser.add_argument(
        "--thurstone_noise",
        type=float,
        default=0.2,
        help="Thurstone noise SD as a multiple of std(visible_score); must be >= 0.",
    )
    parser.add_argument(
        "--mallows_theta",
        type=float,
        default=1.0,
        help=(
            "Mallows concentration parameter (>= 0). 0 gives a uniform "
            "permutation; larger values stay closer to the central ranking."
        ),
    )
    parser.add_argument("--output", type=str, default="synthetic_data.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = generate_dataset(
        n=args.n,
        lambda_1=args.lambda_1,
        boost_strength=args.boost,
        lambda_2=args.lambda_2,
        diversity_penalty_strength=args.diversity_penalty,
        n_categories=args.n_categories,
        seed=args.seed,
        ranking_model=args.ranking_model,
        thurstone_noise=args.thurstone_noise,
        mallows_theta=args.mallows_theta,
    )

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(f"Ranking model: {args.ranking_model}")
    if args.ranking_model == "thurstone":
        print(f"Thurstone noise scale: {args.thurstone_noise}")
    elif args.ranking_model == "mallows":
        print(f"Mallows theta: {args.mallows_theta}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
#python gen_data.py --lambda_1 0.1 --lambda_2 0.1 --output syn.csv
