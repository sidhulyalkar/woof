"""Synthetic compatibility scenarios for sparse and safety-relevant regimes.

These rows are *training augmentation only*. They carry explicit provenance and a
reduced sample weight. Promotion holdouts must remain real/non-synthetic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIMULATION_VERSION = "woof-pair-simulator-v1"
BEHAVIOR_KEYS = ("energy", "sociability", "caution", "excitability", "trainability")
CONTEXTS = {
    "parallel_walk": {"structure": 0.95, "crowding": 0.15, "resource_pressure": 0.05},
    "quiet_open_space": {"structure": 0.75, "crowding": 0.10, "resource_pressure": 0.05},
    "open_play": {"structure": 0.45, "crowding": 0.30, "resource_pressure": 0.15},
    "busy_dog_park": {"structure": 0.20, "crowding": 0.90, "resource_pressure": 0.45},
    "small_shared_space": {"structure": 0.25, "crowding": 0.60, "resource_pressure": 0.70},
}


@dataclass(frozen=True)
class PetState:
    owner_id: str
    pet_id: str
    age_years: float
    energy: float
    sociability: float
    caution: float
    excitability: float
    trainability: float
    social_risk: float
    observation_confidence: float


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pet(rng: random.Random, index: int, rare_risk: bool = False) -> PetState:
    if rare_risk:
        social_risk = rng.uniform(0.62, 0.98)
        caution = rng.uniform(0.55, 0.95)
        excitability = rng.uniform(0.45, 0.95)
    else:
        social_risk = rng.betavariate(1.3, 7.0)
        caution = rng.betavariate(2.0, 2.8)
        excitability = rng.betavariate(2.1, 2.2)
    return PetState(
        owner_id=f"sim-owner-{index:05d}",
        pet_id=f"sim-pet-{index:05d}",
        age_years=rng.uniform(0.6, 14.0),
        energy=rng.betavariate(2.2, 2.0),
        sociability=rng.betavariate(2.5, 1.9),
        caution=caution,
        excitability=excitability,
        trainability=rng.betavariate(2.4, 2.0),
        social_risk=social_risk,
        observation_confidence=rng.uniform(0.55, 0.98),
    )


def scenario_probability(
    a: PetState,
    b: PetState,
    context_name: str,
    prior_positive_rate: float,
    prior_count: int,
    rng: random.Random,
) -> tuple[float, dict[str, float]]:
    context = CONTEXTS[context_name]
    gaps = {key: abs(getattr(a, key) - getattr(b, key)) for key in BEHAVIOR_KEYS}
    behavior_gap = (
        gaps["energy"] * 0.24
        + gaps["sociability"] * 0.30
        + gaps["caution"] * 0.18
        + gaps["excitability"] * 0.18
        + gaps["trainability"] * 0.10
    )
    max_risk = max(a.social_risk, b.social_risk)
    mean_caution = (a.caution + b.caution) / 2.0
    mean_excitability = (a.excitability + b.excitability) / 2.0
    asymmetry = abs(a.sociability - b.sociability) + abs(a.excitability - b.excitability)
    age_gap = min(abs(a.age_years - b.age_years) / 10.0, 1.0)

    # Structure buffers caution while crowding/resource pressure amplify social risk.
    context_stress = (
        context["crowding"] * (0.55 + 0.45 * mean_excitability)
        + context["resource_pressure"] * (0.45 + 0.55 * max_risk)
        + (1.0 - context["structure"]) * (0.35 + 0.65 * mean_caution)
    ) / 3.0
    safety_interaction = max_risk * (0.55 + 0.45 * context_stress)

    prior_signal = 0.0
    if prior_count:
        shrinkage = prior_count / (prior_count + 3.0)
        prior_signal = shrinkage * (prior_positive_rate - 0.5) * 1.5

    latent = 1.8
    latent -= 3.0 * behavior_gap
    latent -= 0.55 * age_gap
    latent -= 2.2 * safety_interaction
    latent -= 0.45 * asymmetry
    latent -= 1.1 * context_stress
    latent += 0.70 * context["structure"]
    latent += prior_signal
    latent += rng.gauss(0.0, 0.42)

    probability = clamp01(sigmoid(latent))
    return probability, {
        "behavior_gap": behavior_gap,
        "max_social_risk": max_risk,
        "context_stress": context_stress,
        "safety_interaction": safety_interaction,
        "age_gap": age_gap,
    }


def generate(
    output: Path,
    manifest_path: Path,
    seed: int,
    pets: int,
    interactions: int,
    rare_safety_fraction: float,
    sample_weight: float,
) -> dict[str, object]:
    if not 0.0 <= rare_safety_fraction <= 0.8:
        raise ValueError("rare_safety_fraction must be between 0 and 0.8")
    if not 0.0 < sample_weight <= 1.0:
        raise ValueError("sample_weight must be in (0, 1]")
    if pets < 20 or interactions < 100:
        raise ValueError("generate at least 20 pets and 100 interactions")

    rng = random.Random(seed)
    rare_count = max(2, int(pets * rare_safety_fraction))
    population = [make_pet(rng, index, index < rare_count) for index in range(pets)]
    pair_history: dict[tuple[str, str], list[int]] = {}
    rows: list[dict[str, object]] = []
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for index in range(interactions):
        force_rare = rng.random() < rare_safety_fraction
        if force_rare:
            a = rng.choice(population[:rare_count])
            b = rng.choice(population)
            while b.pet_id == a.pet_id:
                b = rng.choice(population)
            synthetic_reason = "rare_safety_regime"
        else:
            a, b = rng.sample(population, 2)
            synthetic_reason = "coverage_augmentation"

        if a.pet_id > b.pet_id:
            a, b = b, a
        pair = (a.pet_id, b.pet_id)
        history = pair_history.setdefault(pair, [])
        prior_count = len(history)
        prior_positive_rate = sum(history) / prior_count if prior_count else 0.5

        if force_rare and rng.random() < 0.65:
            context_name = rng.choice(["busy_dog_park", "small_shared_space", "open_play"])
        else:
            context_name = rng.choice(list(CONTEXTS))

        probability, diagnostics = scenario_probability(
            a,
            b,
            context_name,
            prior_positive_rate,
            prior_count,
            rng,
        )
        positive = int(rng.random() < probability)
        history.append(positive)

        if positive:
            rating = rng.choices([3, 4, 5], [0.08, 0.48, 0.44])[0]
            feedback = "great_match|calm_recovery" if rating >= 4 else "neutral"
        else:
            rating = rng.choices([1, 2, 3], [0.20, 0.50, 0.30])[0]
            if diagnostics["max_social_risk"] > 0.65:
                feedback = "too_intense|safety_concern"
            elif diagnostics["behavior_gap"] > 0.35:
                feedback = "energy_mismatch"
            else:
                feedback = "stress_signals"

        context = CONTEXTS[context_name]
        row: dict[str, object] = {
            "outcome_id": f"sim-outcome-{index:07d}",
            "occurred_at": (start + timedelta(hours=index * 3)).isoformat(),
            "owner_a_id": a.owner_id,
            "owner_b_id": b.owner_id,
            "pet_a_id": a.pet_id,
            "pet_b_id": b.pet_id,
            "rating": rating,
            "occurred": 1,
            "feedback_tags": feedback,
            "label": int(rating >= 4),
            "data_source": "synthetic",
            "simulation_version": SIMULATION_VERSION,
            "synthetic_reason": synthetic_reason,
            "sample_weight": round(sample_weight, 4),
            "context_name": context_name,
            "context_structure": round(context["structure"], 4),
            "context_crowding": round(context["crowding"], 4),
            "context_resource_pressure": round(context["resource_pressure"], 4),
            "latent_positive_probability": round(probability, 6),
            "pet_a_age_years": round(a.age_years, 4),
            "pet_b_age_years": round(b.age_years, 4),
            "pet_a_social_risk": round(a.social_risk, 4),
            "pet_b_social_risk": round(b.social_risk, 4),
            "pet_a_observation_confidence": round(a.observation_confidence, 4),
            "pet_b_observation_confidence": round(b.observation_confidence, 4),
            **{f"pet_a_{key}": round(getattr(a, key), 4) for key in BEHAVIOR_KEYS},
            **{f"pet_b_{key}": round(getattr(b, key), 4) for key in BEHAVIOR_KEYS},
        }
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest: dict[str, object] = {
        "schemaVersion": "woof-synthetic-pair-scenarios-v1",
        "simulationVersion": SIMULATION_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "rows": len(rows),
        "pets": pets,
        "rareSafetyFraction": rare_safety_fraction,
        "sampleWeight": sample_weight,
        "sha256": sha256_file(output),
        "policy": {
            "trainingOnly": True,
            "promotionHoldoutAllowed": False,
            "containsProtectedHumanAttributes": False,
            "provenanceRequired": True,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/simulation/artifacts/pair_scenarios.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ml/simulation/artifacts/pair_scenarios.manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--pets", type=int, default=300)
    parser.add_argument("--interactions", type=int, default=6000)
    parser.add_argument("--rare-safety-fraction", type=float, default=0.25)
    parser.add_argument("--sample-weight", type=float, default=0.35)
    args = parser.parse_args()
    manifest = generate(
        args.output,
        args.manifest,
        args.seed,
        args.pets,
        args.interactions,
        args.rare_safety_fraction,
        args.sample_weight,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
