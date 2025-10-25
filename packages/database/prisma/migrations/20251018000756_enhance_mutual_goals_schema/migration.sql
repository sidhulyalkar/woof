/*
  Warnings:

  - Added the required column `end_date` to the `mutual_goals` table without a default value. This is not possible if the table is not empty.
  - Added the required column `start_date` to the `mutual_goals` table without a default value. This is not possible if the table is not empty.
  - Added the required column `target_unit` to the `mutual_goals` table without a default value. This is not possible if the table is not empty.

*/
-- AlterTable
ALTER TABLE "mutual_goals" ADD COLUMN     "best_streak" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN     "completed_days" JSONB NOT NULL DEFAULT '[]',
ADD COLUMN     "current_value" DOUBLE PRECISION NOT NULL DEFAULT 0,
ADD COLUMN     "end_date" TIMESTAMP(3) NOT NULL,
ADD COLUMN     "is_recurring" BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN     "metadata" JSONB,
ADD COLUMN     "reminder_time" TEXT,
ADD COLUMN     "start_date" TIMESTAMP(3) NOT NULL,
ADD COLUMN     "streak_count" INTEGER NOT NULL DEFAULT 0,
ADD COLUMN     "target_unit" TEXT NOT NULL;

-- CreateIndex
CREATE INDEX "mutual_goals_status_idx" ON "mutual_goals"("status");

-- CreateIndex
CREATE INDEX "mutual_goals_end_date_idx" ON "mutual_goals"("end_date");
