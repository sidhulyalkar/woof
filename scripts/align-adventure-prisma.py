from pathlib import Path

SCHEMA = Path('packages/database/prisma/schema.prisma')
text = SCHEMA.read_text()

if 'model CareEvent {' in text:
    raise SystemExit('Adventure Prisma models already exist; refusing to duplicate them.')

user_anchor = '''  mediaAssets              MediaAsset[]
  mediaAlbums              MediaAlbum[]
  mediaExternalReferences  MediaExternalReference[]
'''
user_replacement = user_anchor + '''  careEvents               CareEvent[]
  rewardLedgerEntries      RewardLedger[]
  questInteractions        QuestInteraction[]
'''

pet_anchor = '''  mediaAssets             MediaAsset[]
  mediaAlbums             MediaAlbum[]
  mediaExternalReferences MediaExternalReference[]
'''
pet_replacement = pet_anchor + '''  careEvents              CareEvent[]
  rewardLedgerEntries     RewardLedger[]
  questInteractions       QuestInteraction[]
'''

if text.count(user_anchor) != 1:
    raise SystemExit(f'Expected exactly one User relation anchor, found {text.count(user_anchor)}.')
if text.count(pet_anchor) != 1:
    raise SystemExit(f'Expected exactly one Pet relation anchor, found {text.count(pet_anchor)}.')

text = text.replace(user_anchor, user_replacement, 1)
text = text.replace(pet_anchor, pet_replacement, 1)

models = r'''

// ============================================
// WOOF ADVENTURE SYSTEM
// ============================================

model CareEvent {
  id                 String   @id
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
  @@index([userId, occurredAt(sort: Desc)], map: "care_events_user_id_occurred_at_idx")
  @@index([petId, occurredAt(sort: Desc)], map: "care_events_pet_id_occurred_at_idx")
  @@index([pathway, occurredAt(sort: Desc)], map: "care_events_pathway_occurred_at_idx")
  @@index([eventType], map: "care_events_event_type_idx")
  @@map("care_events")
}

model RewardLedger {
  id            String   @id
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

  @@index([userId, createdAt(sort: Desc)], map: "reward_ledger_user_id_created_at_idx")
  @@index([petId, createdAt(sort: Desc)], map: "reward_ledger_pet_id_created_at_idx")
  @@map("reward_ledger")
}

model QuestInteraction {
  id          String   @id
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

text = text.rstrip() + models + '\n'
SCHEMA.write_text(text)
