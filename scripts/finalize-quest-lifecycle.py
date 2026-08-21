from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"Expected patch target not found in {path}")
    file.write_text(text.replace(old, new, 1))


# Keep the Prisma schema authoritative for tables created by the Adventure migration.
schema_path = Path("packages/database/prisma/schema.prisma")
schema = schema_path.read_text()
replace_once(
    str(schema_path),
    """  mediaAssets              MediaAsset[]\n  mediaAlbums              MediaAlbum[]\n  mediaExternalReferences  MediaExternalReference[]\n""",
    """  mediaAssets              MediaAsset[]\n  mediaAlbums              MediaAlbum[]\n  mediaExternalReferences  MediaExternalReference[]\n  careEvents               CareEvent[]\n  rewardLedgerEntries      RewardLedger[]\n  questInteractions        QuestInteraction[]\n""",
)
replace_once(
    str(schema_path),
    """  mediaAssets             MediaAsset[]\n  mediaAlbums             MediaAlbum[]\n  mediaExternalReferences MediaExternalReference[]\n""",
    """  mediaAssets             MediaAsset[]\n  mediaAlbums             MediaAlbum[]\n  mediaExternalReferences MediaExternalReference[]\n  careEvents              CareEvent[]\n  rewardLedgerEntries     RewardLedger[]\n  questInteractions       QuestInteraction[]\n""",
)

adventure_models = r'''

// ============================================
// WOOF ADVENTURE SYSTEM
// ============================================

model CareEvent {
  id                 String   @id @default(uuid())
  userId             String   @map("user_id")
  petId              String?  @map("pet_id")
  eventType          String   @map("event_type")
  pathway            String
  occurredAt         DateTime @map("occurred_at")
  source             String
  evidenceType       String?  @map("evidence_type")
  evidenceConfidence Float    @default(0.65) @map("evidence_confidence")
  context            Json?
  outcome            Json?
  dedupeKey          String   @map("dedupe_key")
  visibility         String   @default("PRIVATE")
  createdAt          DateTime @default(now()) @map("created_at")

  user   User          @relation(fields: [userId], references: [id], onDelete: Cascade)
  pet    Pet?          @relation(fields: [petId], references: [id], onDelete: SetNull)
  reward RewardLedger?

  @@unique([userId, dedupeKey])
  @@index([userId, occurredAt(sort: Desc)])
  @@index([petId, occurredAt(sort: Desc)])
  @@index([pathway, occurredAt(sort: Desc)])
  @@index([eventType])
  @@map("care_events")
}

model RewardLedger {
  id            String   @id @default(uuid())
  careEventId   String   @unique @map("care_event_id")
  userId        String   @map("user_id")
  petId         String?  @map("pet_id")
  bondXp        Int      @map("bond_xp")
  pathwayXp     Json     @map("pathway_xp")
  policyVersion String   @map("policy_version")
  explanation   String
  createdAt     DateTime @default(now()) @map("created_at")

  careEvent CareEvent @relation(fields: [careEventId], references: [id], onDelete: Cascade)
  user      User      @relation(fields: [userId], references: [id], onDelete: Cascade)
  pet       Pet?      @relation(fields: [petId], references: [id], onDelete: SetNull)

  @@index([userId, createdAt(sort: Desc)])
  @@index([petId, createdAt(sort: Desc)])
  @@map("reward_ledger")
}

model QuestInteraction {
  id          String   @id @default(uuid())
  userId      String   @map("user_id")
  petId       String   @map("pet_id")
  questId     String   @map("quest_id")
  interaction String
  pathway     String
  context     Json?
  createdAt   DateTime @default(now()) @map("created_at")

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
  pet  Pet  @relation(fields: [petId], references: [id], onDelete: Cascade)

  @@unique([userId, petId, questId, interaction])
  @@index([userId, petId, createdAt(sort: Desc)])
  @@index([questId])
  @@map("quest_interactions")
}
'''

schema = schema_path.read_text()
if "model CareEvent {" in schema:
    raise RuntimeError("Adventure Prisma models already exist")
schema_path.write_text(schema.rstrip() + adventure_models + "\n")

# Make interaction logging idempotent and expose the trusted selected quest snapshot.
replace_once(
    "apps/api/src/care-events/care-events.service.ts",
    """    const id = randomUUID();
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO quest_interactions (
        id, user_id, pet_id, quest_id, interaction, pathway, context, created_at
      ) VALUES (
        ${id}, ${input.userId}, ${input.petId}, ${input.questId}, ${input.interaction},
        ${input.pathway}, CAST(${JSON.stringify(input.context ?? {})} AS JSONB), NOW()
      )
    `);
    return { id };
  }

  async getSummary(userId: string, petId?: string): Promise<CareSummary> {
""",
    """    const id = randomUUID();
    const rows = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      INSERT INTO quest_interactions (
        id, user_id, pet_id, quest_id, interaction, pathway, context, created_at
      ) VALUES (
        ${id}, ${input.userId}, ${input.petId}, ${input.questId}, ${input.interaction},
        ${input.pathway}, CAST(${JSON.stringify(input.context ?? {})} AS JSONB), NOW()
      )
      ON CONFLICT (user_id, pet_id, quest_id, interaction)
      DO UPDATE SET
        pathway = EXCLUDED.pathway,
        context = EXCLUDED.context
      RETURNING id
    `);
    return { id: rows[0]?.id ?? id };
  }

  async getRecentSelectedQuestContext(userId: string, petId: string, questId: string) {
    await this.assertOwnedPet(userId, petId);
    const rows = await this.prisma.$queryRaw<
      Array<{ context: Record<string, unknown> | null; created_at: Date }>
    >(Prisma.sql`
      SELECT context, created_at
      FROM quest_interactions
      WHERE user_id = ${userId}
        AND pet_id = ${petId}
        AND quest_id = ${questId}
        AND interaction = 'SELECTED'
        AND created_at >= NOW() - INTERVAL '72 hours'
      LIMIT 1
    `);
    return rows[0] ?? null;
  }

  async getSummary(userId: string, petId?: string): Promise<CareSummary> {
""",
)

# Persist the exact server-approved quest snapshot at selection time and fall back to it
# if the recommendation ranking changes while the owner is out doing the quest.
replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    """      pathway: quest.primaryPathway,
      context: { questKey: quest.key, confidence: quest.confidence },
    });
""",
    """      pathway: quest.primaryPathway,
      context: {
        questKey: quest.key,
        confidence: quest.confidence,
        questSnapshot: quest,
      },
    });
""",
)
replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    """    const dashboard = await this.getDashboard(userId, dto.petId);
    const quest = dashboard.quests.find((candidate) => candidate.id === questId);
    if (!quest) throw new NotFoundException('This quest is no longer available');

    const safeOptOut = Boolean(dto.safeOptOut && quest.safeStopEligible);
""",
    """    const dashboard = await this.getDashboard(userId, dto.petId);
    const currentQuest = dashboard.quests.find((candidate) => candidate.id === questId);
    const selected = currentQuest
      ? null
      : await this.careEvents.getRecentSelectedQuestContext(userId, dto.petId, questId);
    const quest = currentQuest ?? this.questFromSnapshot(selected?.context?.questSnapshot, questId);
    if (!quest) throw new NotFoundException('This quest is no longer available');

    const safeOptOut = Boolean(dto.safeOptOut && quest.safeStopEligible);
""",
)
replace_once(
    "apps/api/src/adventure/adventure.service.ts",
    """  private buildQuests(
""",
    """  private questFromSnapshot(value: unknown, questId: string): Quest | null {
    if (!value || typeof value !== 'object') return null;
    const candidate = value as Partial<Quest>;
    if (
      candidate.id !== questId ||
      typeof candidate.key !== 'string' ||
      typeof candidate.title !== 'string' ||
      typeof candidate.description !== 'string' ||
      typeof candidate.why !== 'string' ||
      typeof candidate.primaryPathway !== 'string' ||
      !Array.isArray(candidate.pathways) ||
      typeof candidate.xp !== 'number' ||
      typeof candidate.confidence !== 'number' ||
      typeof candidate.href !== 'string' ||
      typeof candidate.actionLabel !== 'string' ||
      typeof candidate.safeStopEligible !== 'boolean' ||
      typeof candidate.personalRelevance !== 'number' ||
      typeof candidate.expiresAt !== 'string'
    ) {
      return null;
    }
    return candidate as Quest;
  }

  private buildQuests(
""",
)

# Await selection persistence before leaving Today, then navigate regardless of a network
# failure so logging can never trap the user on the screen.
replace_once(
    "apps/web/src/app/page.tsx",
    """import Link from 'next/link';
import { useState } from 'react';
""",
    """import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
""",
)
replace_once(
    "apps/web/src/app/page.tsx",
    """export default function HomePage() {
  const queryClient = useQueryClient();
""",
    """export default function HomePage() {
  const queryClient = useQueryClient();
  const router = useRouter();
""",
)
replace_once(
    "apps/web/src/app/page.tsx",
    """  const openCompletion = (quest: AdventureQuest, optOut = false) => {
    setClosingQuest(quest);
    setCompletionMessage(null);
    setSafeOptOut(optOut);
    setDogExperience(optOut ? 'not_their_thing' : null);
  };

  return (
""",
    """  const openCompletion = (quest: AdventureQuest, optOut = false) => {
    setClosingQuest(quest);
    setCompletionMessage(null);
    setSafeOptOut(optOut);
    setDogExperience(optOut ? 'not_their_thing' : null);
  };

  const startQuest = async (quest: AdventureQuest) => {
    if (!data) return;
    try {
      await adventureApi.selectQuest(quest.id, data.pet.id);
    } finally {
      router.push(quest.href);
    }
  };

  return (
""",
)
replace_once(
    "apps/web/src/app/page.tsx",
    """                          <Button asChild size="sm">
                            <Link
                              href={quest.href}
                              onClick={() => void adventureApi.selectQuest(quest.id, data.pet.id)}
                            >
                              {quest.actionLabel}
                            </Link>
                          </Button>
""",
    """                          <Button size="sm" onClick={() => void startQuest(quest)}>
                            {quest.actionLabel}
                          </Button>
""",
)
