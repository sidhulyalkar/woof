-- Woof Adventure System v1
-- Trusted domain events are the only source of Bond XP. The client never writes rewards directly.

CREATE TABLE "care_events" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT,
    "event_type" TEXT NOT NULL,
    "pathway" TEXT NOT NULL,
    "occurred_at" TIMESTAMP(3) NOT NULL,
    "source" TEXT NOT NULL,
    "evidence_type" TEXT,
    "evidence_confidence" DOUBLE PRECISION NOT NULL DEFAULT 0.65,
    "context" JSONB,
    "outcome" JSONB,
    "dedupe_key" TEXT NOT NULL,
    "visibility" TEXT NOT NULL DEFAULT 'PRIVATE',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "care_events_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "care_events_pathway_check" CHECK ("pathway" IN ('MOVE', 'EXPLORE', 'ENRICH', 'LEARN', 'CONNECT', 'CARE', 'RECOVER', 'BOND')),
    CONSTRAINT "care_events_confidence_check" CHECK ("evidence_confidence" >= 0 AND "evidence_confidence" <= 1),
    CONSTRAINT "care_events_visibility_check" CHECK ("visibility" IN ('PRIVATE', 'HOUSEHOLD', 'FRIENDS'))
);

CREATE UNIQUE INDEX "care_events_user_id_dedupe_key_key"
    ON "care_events"("user_id", "dedupe_key");
CREATE INDEX "care_events_user_id_occurred_at_idx"
    ON "care_events"("user_id", "occurred_at" DESC);
CREATE INDEX "care_events_pet_id_occurred_at_idx"
    ON "care_events"("pet_id", "occurred_at" DESC);
CREATE INDEX "care_events_pathway_occurred_at_idx"
    ON "care_events"("pathway", "occurred_at" DESC);
CREATE INDEX "care_events_event_type_idx"
    ON "care_events"("event_type");

ALTER TABLE "care_events"
    ADD CONSTRAINT "care_events_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "care_events"
    ADD CONSTRAINT "care_events_pet_id_fkey"
    FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE SET NULL ON UPDATE CASCADE;

CREATE TABLE "reward_ledger" (
    "id" TEXT NOT NULL,
    "care_event_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT,
    "bond_xp" INTEGER NOT NULL,
    "pathway_xp" JSONB NOT NULL,
    "policy_version" TEXT NOT NULL,
    "explanation" TEXT NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "reward_ledger_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "reward_ledger_bond_xp_check" CHECK ("bond_xp" >= 0)
);

CREATE UNIQUE INDEX "reward_ledger_care_event_id_key"
    ON "reward_ledger"("care_event_id");
CREATE INDEX "reward_ledger_user_id_created_at_idx"
    ON "reward_ledger"("user_id", "created_at" DESC);
CREATE INDEX "reward_ledger_pet_id_created_at_idx"
    ON "reward_ledger"("pet_id", "created_at" DESC);

ALTER TABLE "reward_ledger"
    ADD CONSTRAINT "reward_ledger_care_event_id_fkey"
    FOREIGN KEY ("care_event_id") REFERENCES "care_events"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "reward_ledger"
    ADD CONSTRAINT "reward_ledger_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "reward_ledger"
    ADD CONSTRAINT "reward_ledger_pet_id_fkey"
    FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE SET NULL ON UPDATE CASCADE;

CREATE TABLE "quest_interactions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "quest_id" TEXT NOT NULL,
    "interaction" TEXT NOT NULL,
    "pathway" TEXT NOT NULL,
    "context" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "quest_interactions_pkey" PRIMARY KEY ("id"),
    CONSTRAINT "quest_interactions_interaction_check" CHECK ("interaction" IN ('SHOWN', 'SELECTED', 'DISMISSED', 'COMPLETED')),
    CONSTRAINT "quest_interactions_pathway_check" CHECK ("pathway" IN ('MOVE', 'EXPLORE', 'ENRICH', 'LEARN', 'CONNECT', 'CARE', 'RECOVER', 'BOND'))
);

CREATE UNIQUE INDEX "quest_interactions_user_id_pet_id_quest_id_interaction_key"
    ON "quest_interactions"("user_id", "pet_id", "quest_id", "interaction");
CREATE INDEX "quest_interactions_user_pet_created_at_idx"
    ON "quest_interactions"("user_id", "pet_id", "created_at" DESC);
CREATE INDEX "quest_interactions_quest_id_idx"
    ON "quest_interactions"("quest_id");

ALTER TABLE "quest_interactions"
    ADD CONSTRAINT "quest_interactions_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "quest_interactions"
    ADD CONSTRAINT "quest_interactions_pet_id_fkey"
    FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;
