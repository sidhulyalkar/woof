#!/usr/bin/env python3
"""Fail closed if Pack locality regresses to arbitrary or precise location authority."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "packages/database/prisma/migrations/20260911233500_add_structured_pack_locality/migration.sql"
CATALOG = ROOT / "apps/api/src/social-adventure/pack-locality.catalog.ts"
LOCALITY_SERVICE = ROOT / "apps/api/src/social-adventure/pack-locality.service.ts"
DTO = ROOT / "apps/api/src/social-adventure/dto/social-adventure.dto.ts"
CONTROLLER = ROOT / "apps/api/src/social-adventure/social-adventure.controller.ts"
WEB_API = ROOT / "apps/web/src/lib/api/social-adventure.ts"
WEB_PACKS = ROOT / "apps/web/src/app/community/packs/page.tsx"
MOBILE_API = ROOT / "apps/mobile/src/api/social-adventure.ts"
MOBILE_PACKS = ROOT / "apps/mobile/src/screens/PacksScreen.tsx"
MOBILE_VIEW = ROOT / "apps/mobile/src/components/community/SocialAdventurePacksView.tsx"
DOC = ROOT / "docs/NATIVE_SOCIAL_ADVENTURE_V1.md"

CLIENT_REGION_IDS = {
    "us-ca-san-francisco",
    "us-ca-south-bay",
    "us-ca-peninsula",
    "us-ca-east-bay",
    "us-ca-north-bay",
    "us-ca-santa-cruz-county",
}


def fail(message: str) -> None:
    raise SystemExit(message)


def read(path: Path) -> str:
    if not path.is_file():
        fail(f"missing Pack locality authority file: {path.relative_to(ROOT)}")
    return path.read_text()


def require(text: str, label: str, *markers: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        fail(f"{label}: missing required markers: {missing}")


def reject(text: str, label: str, *markers: str) -> None:
    present = [marker for marker in markers if marker in text]
    if present:
        fail(f"{label}: forbidden markers present: {present}")


def main() -> None:
    migration = read(MIGRATION)
    catalog = read(CATALOG)
    locality_service = read(LOCALITY_SERVICE)
    dto = read(DTO)
    controller = read(CONTROLLER)
    web_api = read(WEB_API)
    web_packs = read(WEB_PACKS)
    mobile_api = read(MOBILE_API)
    mobile_packs = read(MOBILE_PACKS)
    mobile_view = read(MOBILE_VIEW)
    doc = read(DOC)

    require(
        migration,
        "structured locality migration",
        "CREATE TABLE IF NOT EXISTS dogos_social.coarse_regions",
        "granularity IN ('METRO', 'COUNTY', 'BROAD_DISTRICT')",
        "UPDATE dogos_social.packs\nSET region_key = NULL\nWHERE scope = 'LOCAL';",
        "REFERENCES dogos_social.coarse_regions(id)",
        "ON UPDATE RESTRICT",
        "ON DELETE RESTRICT",
        "region_key IS NOT NULL",
        "never arbitrary user location text",
    )
    if migration.index("SET region_key = NULL") > migration.index(
        "REFERENCES dogos_social.coarse_regions(id)"
    ):
        fail("legacy free-form locality must be purged before the approved-region foreign key is installed")
    reject(
        migration.lower(),
        "structured locality migration",
        "reverse_geocode",
        "st_point",
        "latitude",
        "longitude",
        "raise notice",
        "raise log",
    )

    catalog_ids = set(re.findall(r"id: '([^']+)'", catalog))
    if catalog_ids != CLIENT_REGION_IDS:
        fail(f"client-selectable coarse-region catalog drifted: {sorted(catalog_ids)}")
    if "test-region" in catalog_ids:
        fail("internal database fixture region leaked into the client-selectable catalog")

    migration_seed_ids = set(re.findall(r"\('([^']+)', '[^']+', 'US'", migration))
    if not CLIENT_REGION_IDS.issubset(migration_seed_ids):
        fail("application region catalog contains IDs absent from the migrated database catalog")
    if "test-region" not in migration_seed_ids:
        fail("reserved direct-SQL test region is missing from database catalog")

    require(
        dto,
        "Pack locality DTO",
        "PACK_COARSE_REGION_IDS",
        "@IsIn([...PACK_COARSE_REGION_IDS])",
        "Server-approved broad locality identity",
        "free-form locality text are rejected",
    )
    reject(dto, "Pack locality DTO", "@Matches(", "Normalize", "normalizeRegionKey")

    require(
        locality_service,
        "Pack locality service",
        "decorateCatalog",
        "pack.joined || this.isApprovedRegionId(pack.regionKey)",
        "LEGACY_UNVERIFIED",
        "requireLocalityAuthority",
        "repairLocality",
        "pack.ownerUserId !== userId",
        "pack.regionKey !== null",
        "region_key IS NULL",
    )
    reject(locality_service, "Pack locality service", "Logger(", "console.log", "regionKey.trim")

    require(
        controller,
        "Pack locality controller",
        "@Get('regions')",
        "@Put('packs/:packId/locality')",
        "await this.packLocality.requireLocalityAuthority(packId)",
        "this.packLocality.decorateCatalog(catalog)",
        "this.packLocality.decoratePack(created)",
    )

    for label, api in [("Web API", web_api), ("native API", mobile_api)]:
        require(
            api,
            label,
            "'/social-adventure/regions'",
            "repairPackLocality",
            "localityStatus: 'APPROVED' | 'LEGACY_UNVERIFIED'",
            "coarseRegion: PackCoarseRegion | null",
        )

    require(
        web_packs,
        "Web Packs",
        "socialAdventureApi.regions",
        "<select",
        "repairPackLocality",
        "LEGACY_UNVERIFIED",
        "coarseRegion?.displayName",
        "Woof discarded the Pack&apos;s old free-form locality",
    )
    require(
        mobile_packs + mobile_view,
        "native Packs",
        "socialAdventureApi.regions()",
        "RegionChoices",
        "repairPackLocality",
        "LEGACY_UNVERIFIED",
        "coarseRegion?.displayName",
        "approved broad area",
    )

    client_surface = web_packs + mobile_packs + mobile_view
    for forbidden in [
        "normalizeRegionKey",
        'placeholder="south-bay-ca"',
        "getCurrentPosition",
        "watchPosition",
        "navigator.geolocation",
        "expo-location",
        "Location.request",
    ]:
        reject(client_surface, "maintained Pack clients", forbidden)

    require(
        doc,
        "Pack locality documentation",
        "server-approved structured coarse-region identity",
        "does not parse, normalize, map, reverse-geocode, or log those values",
        "LEGACY_UNVERIFIED",
        "hidden from nonmember public discovery",
        "new joins fail closed",
        "local standings fail closed",
        "not an anonymity guarantee",
    )

    print(
        "Structured Pack locality authority OK: approved catalog only, legacy free text discarded without inference, "
        "and locality-dependent discovery/join/rank fail closed until repair."
    )


if __name__ == "__main__":
    main()
