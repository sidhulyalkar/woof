"""Generate deterministic, privacy-safe post-meetup outcomes for evaluation.

This dataset is deliberately synthetic. It exists to exercise the same temporal
feature and evaluation pipeline that real, consented beta outcomes will use later.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

BEHAVIOR_KEYS = ["energy", "sociability", "caution", "excitability", "trainability"]


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def generate(output: Path, seed: int, owners: int, interactions: int) -> None:
    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    owner_rows = []
    for index in range(owners):
        behavior = {key: rng.betavariate(2.2, 2.2) for key in BEHAVIOR_KEYS}
        behavior["social_risk"] = rng.betavariate(1.4, 6.5)
        owner_rows.append(
            {
                "owner_id": f"synthetic-owner-{index:04d}",
                "pet_id": f"synthetic-pet-{index:04d}",
                "age_years": round(rng.uniform(0.7, 12.5), 3),
                **behavior,
            }
        )

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    pair_counts: dict[tuple[int, int], int] = {}

    for interaction_index in range(interactions):
        a_index, b_index = rng.sample(range(owners), 2)
        a_index, b_index = sorted((a_index, b_index))
        a = owner_rows[a_index]
        b = owner_rows[b_index]
        pair_key = (a_index, b_index)
        prior_pair_count = pair_counts.get(pair_key, 0)
        pair_counts[pair_key] = prior_pair_count + 1

        behavior_gap = sum(abs(a[key] - b[key]) for key in BEHAVIOR_KEYS) / len(BEHAVIOR_KEYS)
        age_gap = min(abs(a["age_years"] - b["age_years"]) / 10.0, 1.0)
        social_risk = max(a["social_risk"], b["social_risk"])
        latent = 1.7 - 2.7 * behavior_gap - 0.65 * age_gap - 1.8 * social_risk
        latent += min(prior_pair_count, 3) * 0.18
        latent += rng.gauss(0, 0.5)
        positive_probability = 1.0 / (1.0 + math.exp(-latent))
        positive = rng.random() < positive_probability

        if positive:
            rating = rng.choices([3, 4, 5], weights=[0.1, 0.45, 0.45])[0]
            tags = "great_match|owner_friendly" if rating >= 4 else "owner_friendly"
        else:
            rating = rng.choices([1, 2, 3], weights=[0.25, 0.5, 0.25])[0]
            tags = rng.choice(["energy_mismatch", "temperament", "too_intense"])

        occurred_at = start + timedelta(hours=interaction_index * 8 + rng.randint(0, 5))
        rows.append(
            {
                "outcome_id": f"synthetic-outcome-{interaction_index:06d}",
                "occurred_at": occurred_at.isoformat(),
                "owner_a_id": a["owner_id"],
                "owner_b_id": b["owner_id"],
                "pet_a_id": a["pet_id"],
                "pet_b_id": b["pet_id"],
                "rating": rating,
                "occurred": 1,
                "feedback_tags": tags,
                "label": int(rating >= 4),
                "pet_a_age_years": a["age_years"],
                "pet_b_age_years": b["age_years"],
                **{f"pet_a_{key}": round(clamp(a[key]), 4) for key in BEHAVIOR_KEYS},
                **{f"pet_b_{key}": round(clamp(b[key]), 4) for key in BEHAVIOR_KEYS},
                "pet_a_social_risk": round(a["social_risk"], 4),
                "pet_b_social_risk": round(b["social_risk"], 4),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ml/evaluation/artifacts/seeded_outcomes.csv"))
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--owners", type=int, default=180)
    parser.add_argument("--interactions", type=int, default=2400)
    args = parser.parse_args()
    generate(args.output, args.seed, args.owners, args.interactions)
