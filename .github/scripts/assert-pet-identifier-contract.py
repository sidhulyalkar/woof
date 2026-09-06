#!/usr/bin/env python3
"""Keep API DTO pet identifiers compatible with every canonical persisted pet ID."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API_SRC = ROOT / "apps/api/src"
HELPER = API_SRC / "common/validation/pet-identifier.ts"
HELPER_SPEC = API_SRC / "common/validation/pet-identifier.spec.ts"
PETS_SERVICE = API_SRC / "pets/pets.service.ts"

PROPERTY_RE = re.compile(r"^\s*(petId|petIds)(?:\?|!)?\s*:")
ANY_PROPERTY_RE = re.compile(r"^\s*[A-Za-z_$][\w$]*(?:\?|!)?\s*:")
CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+")


def require(path: Path, markers: list[str]) -> None:
    if not path.is_file():
        raise SystemExit(f"required pet-identifier source missing: {path.relative_to(ROOT)}")
    text = path.read_text()
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{path.relative_to(ROOT)} missing pet-identifier markers: {missing}")


def owning_decorator_block(lines: list[str], property_index: int) -> str:
    start = property_index - 1
    while start >= 0:
        line = lines[start]
        if ANY_PROPERTY_RE.match(line) or CLASS_RE.match(line):
            break
        start -= 1
    return "\n".join(lines[start + 1 : property_index])


def validate_dto_pet_fields() -> None:
    failures: list[str] = []
    validated = 0
    for path in sorted(API_SRC.rglob("dto/*.ts")):
        if path.name.endswith(".spec.ts"):
            continue
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            match = PROPERTY_RE.match(line)
            if not match:
                continue
            validated += 1
            decorators = owning_decorator_block(lines, index)
            if "@IsPetIdentifier(" not in decorators:
                failures.append(
                    f"{path.relative_to(ROOT)}:{index + 1} {match.group(1)} must use @IsPetIdentifier"
                )
            if "@IsUUID" in decorators:
                failures.append(
                    f"{path.relative_to(ROOT)}:{index + 1} {match.group(1)} must not retain @IsUUID"
                )
    if validated < 10:
        raise SystemExit("pet-identifier guard found unexpectedly few DTO pet fields")
    if failures:
        raise SystemExit("\n".join(failures))


def main() -> None:
    require(
        HELPER,
        [
            "PET_IDENTIFIER_PATTERN",
            "pet_[0-9a-f]{32}",
            "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "export function IsPetIdentifier",
            "Matches(PET_IDENTIFIER_PATTERN",
        ],
    )
    require(
        HELPER_SPEC,
        [
            "canonical pet identifier validation",
            "pet_0123456789abcdef0123456789abcdef",
            "@IsPetIdentifier()",
            "@IsPetIdentifier({ each: true })",
            "rejects non-canonical pet identifier",
        ],
    )
    require(
        PETS_SERVICE,
        [
            "private replaySafePetId",
            "return `pet_${digest.slice(0, 32)}`",
        ],
    )
    validate_dto_pet_fields()
    print(
        "Pet identifier authority is coherent: DTOs accept UUID-shaped legacy IDs and "
        "replay-safe pet_<32hex> IDs without widening unrelated identifiers."
    )


if __name__ == "__main__":
    main()
