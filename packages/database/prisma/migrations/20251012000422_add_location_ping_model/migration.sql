-- CreateTable
CREATE TABLE "location_pings" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "lat" DOUBLE PRECISION NOT NULL,
    "lng" DOUBLE PRECISION NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "activity_type" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "location_pings_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "location_pings_user_id_idx" ON "location_pings"("user_id");

-- CreateIndex
CREATE INDEX "location_pings_timestamp_idx" ON "location_pings"("timestamp");

-- CreateIndex
CREATE INDEX "location_pings_lat_lng_idx" ON "location_pings"("lat", "lng");
