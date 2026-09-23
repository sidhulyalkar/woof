CREATE TABLE "meetup_outcomes" (
  "id" TEXT NOT NULL,
  "proposal_id" TEXT NOT NULL,
  "participant_id" TEXT NOT NULL,
  "occurred" BOOLEAN NOT NULL,
  "dog_experience" TEXT,
  "owner_experience" TEXT,
  "meet_again" TEXT,
  "rating" INTEGER,
  "feedback_tags" TEXT[] DEFAULT ARRAY[]::TEXT[],
  "checklist_ok" BOOLEAN,
  "notes" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

  CONSTRAINT "meetup_outcomes_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "meetup_outcomes_proposal_id_participant_id_key"
  ON "meetup_outcomes"("proposal_id", "participant_id");
CREATE INDEX "meetup_outcomes_participant_id_idx" ON "meetup_outcomes"("participant_id");
CREATE INDEX "meetup_outcomes_proposal_id_idx" ON "meetup_outcomes"("proposal_id");

ALTER TABLE "meetup_outcomes"
  ADD CONSTRAINT "meetup_outcomes_proposal_id_fkey"
  FOREIGN KEY ("proposal_id") REFERENCES "meetup_proposals"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
