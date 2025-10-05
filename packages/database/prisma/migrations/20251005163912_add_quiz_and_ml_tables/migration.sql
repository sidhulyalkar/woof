-- CreateTable
CREATE TABLE "quiz_responses" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT,
    "session_id" TEXT NOT NULL,
    "responses" JSONB NOT NULL,
    "completed_at" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "quiz_responses_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ml_feature_vectors" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT,
    "features" JSONB NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ml_feature_vectors_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "user_interactions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "target_user_id" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "matched" BOOLEAN NOT NULL DEFAULT false,
    "meetup_occurred" BOOLEAN NOT NULL DEFAULT false,
    "meetup_rating" INTEGER,
    "compatibility_score" DOUBLE PRECISION,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "user_interactions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ml_training_data" (
    "id" TEXT NOT NULL,
    "dataPoint" JSONB NOT NULL,
    "label" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ml_training_data_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "quiz_responses_user_id_idx" ON "quiz_responses"("user_id");

-- CreateIndex
CREATE INDEX "quiz_responses_session_id_idx" ON "quiz_responses"("session_id");

-- CreateIndex
CREATE INDEX "quiz_responses_created_at_idx" ON "quiz_responses"("created_at");

-- CreateIndex
CREATE UNIQUE INDEX "ml_feature_vectors_user_id_key" ON "ml_feature_vectors"("user_id");

-- CreateIndex
CREATE UNIQUE INDEX "ml_feature_vectors_pet_id_key" ON "ml_feature_vectors"("pet_id");

-- CreateIndex
CREATE INDEX "ml_feature_vectors_user_id_idx" ON "ml_feature_vectors"("user_id");

-- CreateIndex
CREATE INDEX "user_interactions_user_id_idx" ON "user_interactions"("user_id");

-- CreateIndex
CREATE INDEX "user_interactions_target_user_id_idx" ON "user_interactions"("target_user_id");

-- CreateIndex
CREATE INDEX "user_interactions_timestamp_idx" ON "user_interactions"("timestamp");

-- CreateIndex
CREATE INDEX "user_interactions_action_idx" ON "user_interactions"("action");

-- CreateIndex
CREATE INDEX "ml_training_data_created_at_idx" ON "ml_training_data"("created_at");

-- AddForeignKey
ALTER TABLE "quiz_responses" ADD CONSTRAINT "quiz_responses_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ml_feature_vectors" ADD CONSTRAINT "ml_feature_vectors_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_interactions" ADD CONSTRAINT "user_interactions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "ml_feature_vectors"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "user_interactions" ADD CONSTRAINT "user_interactions_target_user_id_fkey" FOREIGN KEY ("target_user_id") REFERENCES "ml_feature_vectors"("user_id") ON DELETE CASCADE ON UPDATE CASCADE;
