-- Adaptive Adventure 1B: append-only, household/pet-scoped personalization evidence.
-- The pair foreign key prevents profile rows from referring to a pet outside the
-- declared household. Durable profile evidence and question-response history are
-- intentionally separate so cooldown/skip semantics never become preference labels.

CREATE TABLE "adaptive_profile_evidence" (
    "id" TEXT NOT NULL,
    "household_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "dimension" TEXT NOT NULL,
    "subject" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "value" JSONB,
    "confidence" DOUBLE PRECISION NOT NULL,
    "provenance" TEXT NOT NULL,
    "schema_version" TEXT NOT NULL DEFAULT 'adaptive-profile-v1',
    "source_user_id" TEXT,
    "occurred_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "adaptive_profile_evidence_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "adaptive_profile_question_responses" (
    "id" TEXT NOT NULL,
    "household_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "question_id" TEXT NOT NULL,
    "policy_version" TEXT NOT NULL,
    "outcome" TEXT NOT NULL,
    "answer" JSONB,
    "asked_at" TIMESTAMP(3) NOT NULL,
    "responded_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "adaptive_profile_question_responses_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "adaptive_profile_evidence_pair_dimension_occurred_at_idx"
    ON "adaptive_profile_evidence"("household_id", "pet_id", "dimension", "occurred_at" DESC);

CREATE INDEX "adaptive_profile_evidence_source_user_occurred_at_idx"
    ON "adaptive_profile_evidence"("source_user_id", "occurred_at" DESC);

CREATE INDEX "adaptive_profile_question_pair_asked_at_idx"
    ON "adaptive_profile_question_responses"("household_id", "pet_id", "asked_at" DESC);

CREATE INDEX "adaptive_profile_question_user_responded_at_idx"
    ON "adaptive_profile_question_responses"("user_id", "responded_at" DESC);

ALTER TABLE "adaptive_profile_evidence"
    ADD CONSTRAINT "adaptive_profile_evidence_household_id_pet_id_fkey"
    FOREIGN KEY ("household_id", "pet_id")
    REFERENCES "household_pets"("household_id", "pet_id")
    ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "adaptive_profile_evidence"
    ADD CONSTRAINT "adaptive_profile_evidence_source_user_id_fkey"
    FOREIGN KEY ("source_user_id") REFERENCES "users"("id")
    ON DELETE SET NULL ON UPDATE CASCADE;

ALTER TABLE "adaptive_profile_question_responses"
    ADD CONSTRAINT "adaptive_profile_question_responses_household_id_pet_id_fkey"
    FOREIGN KEY ("household_id", "pet_id")
    REFERENCES "household_pets"("household_id", "pet_id")
    ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "adaptive_profile_question_responses"
    ADD CONSTRAINT "adaptive_profile_question_responses_user_id_fkey"
    FOREIGN KEY ("user_id") REFERENCES "users"("id")
    ON DELETE CASCADE ON UPDATE CASCADE;
