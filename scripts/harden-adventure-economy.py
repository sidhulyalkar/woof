from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"Expected patch target not found in {path}")
    file.write_text(text.replace(old, new, 1))


replace_once(
    "apps/api/src/care-events/care-events.service.ts",
    """    return this.prisma.$transaction(async (tx) => {\n      const existing = await tx.$queryRaw<EventRow[]>(Prisma.sql`\n""",
    """    return this.prisma.$transaction(async (tx) => {\n      // Serialize reward issuance for one user so concurrent legitimate requests cannot\n      // race the daily/pathway caps or a shared dedupe key. The lock lives only for\n      // this transaction and does not block rewards for other users.\n      await tx.$queryRaw(\n        Prisma.sql`SELECT pg_advisory_xact_lock(hashtextextended(${input.userId}, 0))`\n      );\n\n      const existing = await tx.$queryRaw<EventRow[]>(Prisma.sql`\n""",
)

replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    """    const rewardPathway: WellbeingPathway =\n      safeOptOut || learnedMismatch ? 'BOND' : quest.primaryPathway;\n\n    const receipt = await this.careEvents.record({\n""",
    """    const rewardPathway: WellbeingPathway =\n      safeOptOut || learnedMismatch ? 'BOND' : quest.primaryPathway;\n\n    // Memory bonuses require a real, completed private asset owned by this exact\n    // dog-owner pair. A random client-supplied asset ID never changes rewards.\n    const verifiedMemory = dto.memoryAssetId\n      ? await this.prisma.mediaAsset.findFirst({\n          where: {\n            id: dto.memoryAssetId,\n            ownerId: userId,\n            petId: dto.petId,\n            status: 'READY',\n          },\n          select: { id: true },\n        })\n      : null;\n\n    const receipt = await this.careEvents.record({\n""",
)

replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    """        personalRelevance: quest.personalRelevance,\n        newPlace: dto.newPlace ?? false,\n        memoryAdded: Boolean(dto.memoryAssetId),\n        memoryAssetId: dto.memoryAssetId ?? null,\n""",
    """        personalRelevance: quest.personalRelevance,\n        memoryAdded: Boolean(verifiedMemory),\n        memoryAssetId: verifiedMemory?.id ?? null,\n""",
)

replace_once(
    "apps/web/src/app/page.tsx",
    """                            +{quest.xp} XP\n""",
    """                            Base {quest.xp} XP\n""",
)
