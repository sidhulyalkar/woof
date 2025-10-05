-- CreateTable
CREATE TABLE "meetup_proposals" (
    "id" TEXT NOT NULL,
    "proposer_id" TEXT NOT NULL,
    "recipient_id" TEXT NOT NULL,
    "suggested_time" TIMESTAMP(3) NOT NULL,
    "suggested_venue" JSONB NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "occurred_at" TIMESTAMP(3),
    "rating" INTEGER,
    "feedback_tags" TEXT[],
    "checklist_ok" BOOLEAN NOT NULL DEFAULT false,
    "notes" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "meetup_proposals_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "co_activity_segments" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "other_user_id" TEXT,
    "other_pet_id" TEXT,
    "start_time" TIMESTAMP(3) NOT NULL,
    "end_time" TIMESTAMP(3) NOT NULL,
    "distance_m" DOUBLE PRECISION NOT NULL,
    "gps_overlap_m" DOUBLE PRECISION,
    "avg_pace" DOUBLE PRECISION,
    "venue_type" TEXT,
    "gps_trace_ref" TEXT,
    "steps" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "co_activity_segments_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "community_events" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "description" TEXT,
    "host_user_id" TEXT NOT NULL,
    "venue_type" TEXT NOT NULL,
    "lat" DOUBLE PRECISION NOT NULL,
    "lng" DOUBLE PRECISION NOT NULL,
    "venue_name" TEXT,
    "address" TEXT,
    "start_time" TIMESTAMP(3) NOT NULL,
    "end_time" TIMESTAMP(3) NOT NULL,
    "capacity" INTEGER,
    "rsvp_count" INTEGER NOT NULL DEFAULT 0,
    "visibility" TEXT NOT NULL DEFAULT 'PUBLIC',
    "recurring" BOOLEAN NOT NULL DEFAULT false,
    "recurring_pattern" TEXT,
    "post_feedback_score" DOUBLE PRECISION,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "community_events_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "event_rsvps" (
    "id" TEXT NOT NULL,
    "event_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "pet_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'YES',
    "checked_in_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "event_rsvps_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "event_feedback" (
    "id" TEXT NOT NULL,
    "event_id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "vibe_score" INTEGER NOT NULL,
    "pet_density" TEXT,
    "surface_type" TEXT,
    "crowding" TEXT,
    "noise_level" TEXT,
    "tags" TEXT[],
    "notes" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "event_feedback_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "businesses" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "lat" DOUBLE PRECISION NOT NULL,
    "lng" DOUBLE PRECISION NOT NULL,
    "address" TEXT,
    "phone" TEXT,
    "website" TEXT,
    "rating" DOUBLE PRECISION,
    "review_count" INTEGER,
    "hours" JSONB,
    "pet_policies" JSONB,
    "partnered" BOOLEAN NOT NULL DEFAULT false,
    "partner_tier" TEXT,
    "logo_url" TEXT,
    "photos" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "amenities" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "businesses_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "service_intents" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "business_id" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "conversion_followup" BOOLEAN,
    "followup_asked_at" TIMESTAMP(3),
    "followup_response" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "service_intents_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "gamification" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "points" INTEGER NOT NULL DEFAULT 0,
    "level" INTEGER NOT NULL DEFAULT 1,
    "badges" TEXT[],
    "weekly_streak" INTEGER NOT NULL DEFAULT 0,
    "last_active_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "streak_start" TIMESTAMP(3),
    "total_meetups" INTEGER NOT NULL DEFAULT 0,
    "total_events" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "gamification_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "point_transactions" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "points" INTEGER NOT NULL,
    "reason" TEXT NOT NULL,
    "reference_id" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "point_transactions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "proactive_nudges" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "target_user_id" TEXT,
    "type" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "sent_via" TEXT NOT NULL DEFAULT 'push',
    "accepted" BOOLEAN,
    "responded_at" TIMESTAMP(3),
    "sent_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "proactive_nudges_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "nudge_cooldowns" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "target_user_id" TEXT,
    "nudge_type" TEXT NOT NULL,
    "cooldown_until" TIMESTAMP(3) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "nudge_cooldowns_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "safety_verifications" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "vaccine_doc_url" TEXT,
    "vaccine_verified" BOOLEAN NOT NULL DEFAULT false,
    "vaccine_expiry" TIMESTAMP(3),
    "id_doc_url" TEXT,
    "id_verified" BOOLEAN NOT NULL DEFAULT false,
    "trusted_badge" BOOLEAN NOT NULL DEFAULT false,
    "verification_notes" TEXT,
    "verified_at" TIMESTAMP(3),
    "verified_by" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "safety_verifications_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "report_flags" (
    "id" TEXT NOT NULL,
    "reporter_id" TEXT NOT NULL,
    "reported_id" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "description" TEXT,
    "evidence" TEXT[],
    "status" TEXT NOT NULL DEFAULT 'pending',
    "reviewed_by" TEXT,
    "reviewed_at" TIMESTAMP(3),
    "action_taken" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "report_flags_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "blocked_users" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "blocked_id" TEXT NOT NULL,
    "reason" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "blocked_users_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "meetup_proposals_proposer_id_idx" ON "meetup_proposals"("proposer_id");

-- CreateIndex
CREATE INDEX "meetup_proposals_recipient_id_idx" ON "meetup_proposals"("recipient_id");

-- CreateIndex
CREATE INDEX "meetup_proposals_status_idx" ON "meetup_proposals"("status");

-- CreateIndex
CREATE INDEX "meetup_proposals_suggested_time_idx" ON "meetup_proposals"("suggested_time");

-- CreateIndex
CREATE INDEX "co_activity_segments_user_id_idx" ON "co_activity_segments"("user_id");

-- CreateIndex
CREATE INDEX "co_activity_segments_pet_id_idx" ON "co_activity_segments"("pet_id");

-- CreateIndex
CREATE INDEX "co_activity_segments_start_time_idx" ON "co_activity_segments"("start_time");

-- CreateIndex
CREATE INDEX "co_activity_segments_venue_type_idx" ON "co_activity_segments"("venue_type");

-- CreateIndex
CREATE INDEX "community_events_host_user_id_idx" ON "community_events"("host_user_id");

-- CreateIndex
CREATE INDEX "community_events_start_time_idx" ON "community_events"("start_time");

-- CreateIndex
CREATE INDEX "community_events_lat_lng_idx" ON "community_events"("lat", "lng");

-- CreateIndex
CREATE INDEX "community_events_visibility_idx" ON "community_events"("visibility");

-- CreateIndex
CREATE INDEX "event_rsvps_event_id_idx" ON "event_rsvps"("event_id");

-- CreateIndex
CREATE INDEX "event_rsvps_user_id_idx" ON "event_rsvps"("user_id");

-- CreateIndex
CREATE INDEX "event_rsvps_status_idx" ON "event_rsvps"("status");

-- CreateIndex
CREATE UNIQUE INDEX "event_rsvps_event_id_user_id_key" ON "event_rsvps"("event_id", "user_id");

-- CreateIndex
CREATE INDEX "event_feedback_event_id_idx" ON "event_feedback"("event_id");

-- CreateIndex
CREATE INDEX "event_feedback_vibe_score_idx" ON "event_feedback"("vibe_score");

-- CreateIndex
CREATE UNIQUE INDEX "event_feedback_event_id_user_id_key" ON "event_feedback"("event_id", "user_id");

-- CreateIndex
CREATE INDEX "businesses_type_idx" ON "businesses"("type");

-- CreateIndex
CREATE INDEX "businesses_lat_lng_idx" ON "businesses"("lat", "lng");

-- CreateIndex
CREATE INDEX "businesses_partnered_idx" ON "businesses"("partnered");

-- CreateIndex
CREATE INDEX "businesses_rating_idx" ON "businesses"("rating");

-- CreateIndex
CREATE INDEX "service_intents_user_id_idx" ON "service_intents"("user_id");

-- CreateIndex
CREATE INDEX "service_intents_business_id_idx" ON "service_intents"("business_id");

-- CreateIndex
CREATE INDEX "service_intents_action_idx" ON "service_intents"("action");

-- CreateIndex
CREATE INDEX "service_intents_created_at_idx" ON "service_intents"("created_at");

-- CreateIndex
CREATE INDEX "service_intents_conversion_followup_idx" ON "service_intents"("conversion_followup");

-- CreateIndex
CREATE UNIQUE INDEX "gamification_user_id_key" ON "gamification"("user_id");

-- CreateIndex
CREATE INDEX "gamification_points_idx" ON "gamification"("points");

-- CreateIndex
CREATE INDEX "gamification_level_idx" ON "gamification"("level");

-- CreateIndex
CREATE INDEX "gamification_weekly_streak_idx" ON "gamification"("weekly_streak");

-- CreateIndex
CREATE INDEX "point_transactions_user_id_idx" ON "point_transactions"("user_id");

-- CreateIndex
CREATE INDEX "point_transactions_created_at_idx" ON "point_transactions"("created_at");

-- CreateIndex
CREATE INDEX "point_transactions_reason_idx" ON "point_transactions"("reason");

-- CreateIndex
CREATE INDEX "proactive_nudges_user_id_idx" ON "proactive_nudges"("user_id");

-- CreateIndex
CREATE INDEX "proactive_nudges_type_idx" ON "proactive_nudges"("type");

-- CreateIndex
CREATE INDEX "proactive_nudges_sent_at_idx" ON "proactive_nudges"("sent_at");

-- CreateIndex
CREATE INDEX "proactive_nudges_accepted_idx" ON "proactive_nudges"("accepted");

-- CreateIndex
CREATE INDEX "nudge_cooldowns_user_id_idx" ON "nudge_cooldowns"("user_id");

-- CreateIndex
CREATE INDEX "nudge_cooldowns_cooldown_until_idx" ON "nudge_cooldowns"("cooldown_until");

-- CreateIndex
CREATE UNIQUE INDEX "nudge_cooldowns_user_id_target_user_id_nudge_type_key" ON "nudge_cooldowns"("user_id", "target_user_id", "nudge_type");

-- CreateIndex
CREATE UNIQUE INDEX "safety_verifications_user_id_key" ON "safety_verifications"("user_id");

-- CreateIndex
CREATE INDEX "safety_verifications_vaccine_verified_idx" ON "safety_verifications"("vaccine_verified");

-- CreateIndex
CREATE INDEX "safety_verifications_trusted_badge_idx" ON "safety_verifications"("trusted_badge");

-- CreateIndex
CREATE INDEX "report_flags_reported_id_idx" ON "report_flags"("reported_id");

-- CreateIndex
CREATE INDEX "report_flags_status_idx" ON "report_flags"("status");

-- CreateIndex
CREATE INDEX "report_flags_created_at_idx" ON "report_flags"("created_at");

-- CreateIndex
CREATE INDEX "blocked_users_user_id_idx" ON "blocked_users"("user_id");

-- CreateIndex
CREATE INDEX "blocked_users_blocked_id_idx" ON "blocked_users"("blocked_id");

-- CreateIndex
CREATE UNIQUE INDEX "blocked_users_user_id_blocked_id_key" ON "blocked_users"("user_id", "blocked_id");

-- AddForeignKey
ALTER TABLE "event_rsvps" ADD CONSTRAINT "event_rsvps_event_id_fkey" FOREIGN KEY ("event_id") REFERENCES "community_events"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "event_feedback" ADD CONSTRAINT "event_feedback_event_id_fkey" FOREIGN KEY ("event_id") REFERENCES "community_events"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "service_intents" ADD CONSTRAINT "service_intents_business_id_fkey" FOREIGN KEY ("business_id") REFERENCES "businesses"("id") ON DELETE CASCADE ON UPDATE CASCADE;
