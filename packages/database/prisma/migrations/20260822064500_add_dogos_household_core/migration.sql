-- dogOS household + multi-pet activity foundation
-- Additive-only migration. Existing owner/pet/activity columns are retained for
-- rolling compatibility while dogOS moves to household-scoped participation.
--
-- Prisma String @default(uuid()) is stored as TEXT in this schema unless the
-- field is explicitly annotated @db.Uuid. Keep all dogOS identifiers and
-- foreign keys aligned with the existing users/pets/activities TEXT ids.

CREATE TABLE "households" (
  "id" TEXT NOT NULL,
  "name" TEXT NOT NULL DEFAULT 'My household',
  "timezone" TEXT,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" TIMESTAMP(3) NOT NULL,
  CONSTRAINT "households_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "household_members" (
  "id" TEXT NOT NULL,
  "household_id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "role" TEXT NOT NULL DEFAULT 'MEMBER',
  "status" TEXT NOT NULL DEFAULT 'ACTIVE',
  "joined_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "household_members_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "household_pets" (
  "id" TEXT NOT NULL,
  "household_id" TEXT NOT NULL,
  "pet_id" TEXT NOT NULL,
  "status" TEXT NOT NULL DEFAULT 'ACTIVE',
  "joined_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "household_pets_pkey" PRIMARY KEY ("id")
);

ALTER TABLE "activities" ADD COLUMN "household_id" TEXT;

CREATE TABLE "activity_human_participants" (
  "id" TEXT NOT NULL,
  "activity_id" TEXT NOT NULL,
  "user_id" TEXT NOT NULL,
  "role" TEXT NOT NULL DEFAULT 'PARTICIPANT',
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "activity_human_participants_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "activity_pet_participants" (
  "id" TEXT NOT NULL,
  "activity_id" TEXT NOT NULL,
  "pet_id" TEXT NOT NULL,
  "metrics" JSONB,
  "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "activity_pet_participants_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "household_members_household_id_user_id_key"
  ON "household_members"("household_id", "user_id");
CREATE INDEX "household_members_user_id_status_idx"
  ON "household_members"("user_id", "status");

CREATE UNIQUE INDEX "household_pets_household_id_pet_id_key"
  ON "household_pets"("household_id", "pet_id");
CREATE INDEX "household_pets_pet_id_status_idx"
  ON "household_pets"("pet_id", "status");

CREATE INDEX "activities_household_id_started_at_idx"
  ON "activities"("household_id", "started_at");

CREATE UNIQUE INDEX "activity_human_participants_activity_id_user_id_key"
  ON "activity_human_participants"("activity_id", "user_id");
CREATE INDEX "activity_human_participants_user_id_idx"
  ON "activity_human_participants"("user_id");

CREATE UNIQUE INDEX "activity_pet_participants_activity_id_pet_id_key"
  ON "activity_pet_participants"("activity_id", "pet_id");
CREATE INDEX "activity_pet_participants_pet_id_idx"
  ON "activity_pet_participants"("pet_id");

ALTER TABLE "household_members"
  ADD CONSTRAINT "household_members_household_id_fkey"
  FOREIGN KEY ("household_id") REFERENCES "households"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "household_members"
  ADD CONSTRAINT "household_members_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "household_pets"
  ADD CONSTRAINT "household_pets_household_id_fkey"
  FOREIGN KEY ("household_id") REFERENCES "households"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "household_pets"
  ADD CONSTRAINT "household_pets_pet_id_fkey"
  FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "activities"
  ADD CONSTRAINT "activities_household_id_fkey"
  FOREIGN KEY ("household_id") REFERENCES "households"("id") ON DELETE SET NULL ON UPDATE CASCADE;

ALTER TABLE "activity_human_participants"
  ADD CONSTRAINT "activity_human_participants_activity_id_fkey"
  FOREIGN KEY ("activity_id") REFERENCES "activities"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "activity_human_participants"
  ADD CONSTRAINT "activity_human_participants_user_id_fkey"
  FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "activity_pet_participants"
  ADD CONSTRAINT "activity_pet_participants_activity_id_fkey"
  FOREIGN KEY ("activity_id") REFERENCES "activities"("id") ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "activity_pet_participants"
  ADD CONSTRAINT "activity_pet_participants_pet_id_fkey"
  FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Every existing account receives one deterministic personal household.
-- Cast the MD5 through uuid only to produce the same hyphenated UUID-shaped text
-- emitted by HouseholdsService.deterministicUuid; the stored database type is TEXT.
INSERT INTO "households" ("id", "name", "created_at", "updated_at")
SELECT md5('dogos-household:' || u."id")::uuid::text,
       'My household',
       CURRENT_TIMESTAMP,
       CURRENT_TIMESTAMP
FROM "users" u
ON CONFLICT ("id") DO NOTHING;

INSERT INTO "household_members" ("id", "household_id", "user_id", "role", "status", "joined_at")
SELECT md5('dogos-household-member:' || u."id")::uuid::text,
       md5('dogos-household:' || u."id")::uuid::text,
       u."id",
       'OWNER',
       'ACTIVE',
       CURRENT_TIMESTAMP
FROM "users" u
ON CONFLICT ("household_id", "user_id") DO NOTHING;

INSERT INTO "household_pets" ("id", "household_id", "pet_id", "status", "joined_at")
SELECT md5('dogos-household-pet:' || p."id")::uuid::text,
       md5('dogos-household:' || p."owner_id")::uuid::text,
       p."id",
       'ACTIVE',
       CURRENT_TIMESTAMP
FROM "pets" p
ON CONFLICT ("household_id", "pet_id") DO NOTHING;

UPDATE "activities" a
SET "household_id" = md5('dogos-household:' || a."user_id")::uuid::text
WHERE a."household_id" IS NULL;

INSERT INTO "activity_human_participants" ("id", "activity_id", "user_id", "role", "created_at")
SELECT md5('dogos-activity-human:' || a."id" || ':' || a."user_id")::uuid::text,
       a."id",
       a."user_id",
       'RECORDER',
       a."created_at"
FROM "activities" a
ON CONFLICT ("activity_id", "user_id") DO NOTHING;

INSERT INTO "activity_pet_participants" ("id", "activity_id", "pet_id", "metrics", "created_at")
SELECT md5('dogos-activity-pet:' || a."id" || ':' || a."pet_id")::uuid::text,
       a."id",
       a."pet_id",
       a."pet_metrics",
       a."created_at"
FROM "activities" a
WHERE a."pet_id" IS NOT NULL
ON CONFLICT ("activity_id", "pet_id") DO NOTHING;
