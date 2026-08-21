from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"Expected patch target not found in {path}")
    file.write_text(text.replace(old, new, 1))


# Keep the Prisma datamodel authoritative for the additive Adventure tables.
schema_path = Path("packages/database/prisma/schema.prisma")
schema = schema_path.read_text()
if "model CareEvent {" not in schema:
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

  @@unique([userId, dedupeKey], map: "care_events_user_id_dedupe_key_key")
  @@index([userId, occurredAt(sort: Desc)], map: "care_events_user_occurred_at_idx")
  @@index([petId, occurredAt(sort: Desc)], map: "care_events_pet_occurred_at_idx")
  @@index([pathway, occurredAt(sort: Desc)], map: "care_events_pathway_occurred_at_idx")
  @@index([eventType], map: "care_events_event_type_idx")
  @@map("care_events")
}

model RewardLedger {
  id            String   @id @default(uuid())
  careEventId   String   @unique(map: "reward_ledger_care_event_id_key") @map("care_event_id")
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

  @@index([userId, createdAt(sort: Desc)], map: "reward_ledger_user_created_at_idx")
  @@index([petId, createdAt(sort: Desc)], map: "reward_ledger_pet_created_at_idx")
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

  @@unique([userId, petId, questId, interaction], map: "quest_interactions_user_id_pet_id_quest_id_interaction_key")
  @@index([userId, petId, createdAt(sort: Desc)], map: "quest_interactions_user_pet_created_at_idx")
  @@index([questId], map: "quest_interactions_quest_id_idx")
  @@map("quest_interactions")
}
'''
    schema = schema_path.read_text()
    schema_path.write_text(schema.rstrip() + adventure_models + "\n")

# The Adventure service test fixture must match the real Insights pet projection.
replace_once(
    "apps/api/src/adventure/adventure.service.spec.ts",
    "pet: { id: 'pet-1', name: 'Shasta', species: 'DOG' },",
    "pet: { id: 'pet-1', name: 'Shasta', species: 'DOG', avatarUrl: null },",
)

# Narrow nullable telemetry rows before the type predicate. Without the explicit
# map result union TypeScript infers ownerFeedback as required-with-undefined,
# which is not assignable to the domain type where ownerFeedback is optional.
replace_once(
    "apps/api/src/behavior-vision/behavior-vision.service.ts",
    """    return entries
      .map((entry) => {
""",
    """    return entries
      .map<StoredBehaviorObservation | null>((entry) => {
""",
)

# Nudge creation is actor-scoped; pass the authenticated sender explicitly.
replace_once(
    "apps/api/src/chat/chat.gateway.ts",
    "await this.nudgesService.checkChatActivityNudges(data.conversationId).catch((err) => {",
    "await this.nudgesService.checkChatActivityNudges(data.conversationId, userId).catch((err) => {",
)

# Goals was reading a non-existent userId field from the JWT request. Reuse the
# shared authenticated request contract and the strategy's canonical `sub` id.
goals_path = Path("apps/api/src/goals/goals.controller.ts")
goals = goals_path.read_text()
if "AuthenticatedRequest" not in goals:
    goals = goals.replace(
        "import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';",
        "import type { AuthenticatedRequest } from '../auth/authenticated-request';\nimport { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';",
    )
goals = goals.replace("@Request() req,", "@Request() req: AuthenticatedRequest,")
goals = goals.replace("@Request() req)", "@Request() req: AuthenticatedRequest)")
goals = goals.replace("req.user.userId", "req.user.sub")
goals_path.write_text(goals)
