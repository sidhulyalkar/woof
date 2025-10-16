/*
  Warnings:

  - You are about to drop the column `reference_id` on the `point_transactions` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "point_transactions" DROP COLUMN "reference_id",
ADD COLUMN     "related_entity_id" TEXT;

-- AlterTable
ALTER TABLE "users" ADD COLUMN     "total_points" INTEGER NOT NULL DEFAULT 0;

-- CreateTable
CREATE TABLE "badge_awards" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "badge_type" TEXT NOT NULL,
    "awarded_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "badge_awards_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "weekly_streaks" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "current_week" INTEGER NOT NULL DEFAULT 0,
    "last_activity_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "weekly_streaks_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "badge_awards_user_id_idx" ON "badge_awards"("user_id");

-- CreateIndex
CREATE INDEX "badge_awards_badge_type_idx" ON "badge_awards"("badge_type");

-- CreateIndex
CREATE UNIQUE INDEX "badge_awards_user_id_badge_type_key" ON "badge_awards"("user_id", "badge_type");

-- CreateIndex
CREATE UNIQUE INDEX "weekly_streaks_user_id_key" ON "weekly_streaks"("user_id");

-- CreateIndex
CREATE INDEX "weekly_streaks_user_id_idx" ON "weekly_streaks"("user_id");

-- AddForeignKey
ALTER TABLE "community_events" ADD CONSTRAINT "community_events_host_user_id_fkey" FOREIGN KEY ("host_user_id") REFERENCES "users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "event_rsvps" ADD CONSTRAINT "event_rsvps_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "event_feedback" ADD CONSTRAINT "event_feedback_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;
