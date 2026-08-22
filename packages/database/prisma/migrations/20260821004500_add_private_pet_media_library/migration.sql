-- Dedicated private pet-media catalog.
-- Raw media remains in private S3/R2 object storage; PostgreSQL owns metadata,
-- relations, provenance, lifecycle state, and derivative identities.

CREATE TABLE "media_assets" (
    "id" TEXT NOT NULL,
    "owner_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "storage_key" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "mime_type" TEXT NOT NULL,
    "media_type" TEXT NOT NULL,
    "size_bytes" BIGINT NOT NULL,
    "captured_at" TIMESTAMP(3),
    "source" TEXT NOT NULL,
    "provider" TEXT,
    "provider_item_id" TEXT,
    "favorite" BOOLEAN NOT NULL DEFAULT false,
    "status" TEXT NOT NULL DEFAULT 'PENDING',
    "created_from" TEXT NOT NULL,
    "sha256" TEXT,
    "width" INTEGER,
    "height" INTEGER,
    "duration_ms" INTEGER,
    "upload_expires_at" TIMESTAMP(3),
    "completed_at" TIMESTAMP(3),
    "tags" JSONB NOT NULL DEFAULT '[]',
    "linked_observation_ids" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "media_assets_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "media_albums" (
    "id" TEXT NOT NULL,
    "owner_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "cover_asset_id" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "media_albums_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "media_album_assets" (
    "album_id" TEXT NOT NULL,
    "asset_id" TEXT NOT NULL,
    "added_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "media_album_assets_pkey" PRIMARY KEY ("album_id", "asset_id")
);

CREATE TABLE "media_derivatives" (
    "id" TEXT NOT NULL,
    "asset_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "processor_version" TEXT NOT NULL,
    "storage_key" TEXT NOT NULL,
    "mime_type" TEXT,
    "size_bytes" BIGINT,
    "status" TEXT NOT NULL DEFAULT 'PENDING',
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "media_derivatives_pkey" PRIMARY KEY ("id")
);

CREATE TABLE "media_external_references" (
    "id" TEXT NOT NULL,
    "asset_id" TEXT NOT NULL,
    "owner_id" TEXT NOT NULL,
    "pet_id" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "provider_item_id" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "media_external_references_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "media_assets_storage_key_key" ON "media_assets"("storage_key");
CREATE INDEX "media_assets_owner_id_created_at_idx" ON "media_assets"("owner_id", "created_at" DESC);
CREATE INDEX "media_assets_pet_id_captured_at_idx" ON "media_assets"("pet_id", "captured_at" DESC);
CREATE INDEX "media_assets_pet_id_media_type_captured_at_idx" ON "media_assets"("pet_id", "media_type", "captured_at" DESC);
CREATE INDEX "media_assets_provider_provider_item_id_idx" ON "media_assets"("provider", "provider_item_id");
CREATE INDEX "media_assets_sha256_idx" ON "media_assets"("sha256");
CREATE INDEX "media_assets_status_created_at_idx" ON "media_assets"("status", "created_at");
CREATE INDEX "media_albums_owner_id_pet_id_created_at_idx" ON "media_albums"("owner_id", "pet_id", "created_at");
CREATE INDEX "media_album_assets_asset_id_idx" ON "media_album_assets"("asset_id");
CREATE UNIQUE INDEX "media_derivatives_asset_id_kind_processor_version_key" ON "media_derivatives"("asset_id", "kind", "processor_version");
CREATE UNIQUE INDEX "media_derivatives_storage_key_key" ON "media_derivatives"("storage_key");
CREATE INDEX "media_derivatives_status_created_at_idx" ON "media_derivatives"("status", "created_at");
CREATE INDEX "media_external_references_owner_id_provider_provider_item_id_idx" ON "media_external_references"("owner_id", "provider", "provider_item_id");
CREATE INDEX "media_external_references_pet_id_provider_idx" ON "media_external_references"("pet_id", "provider");

ALTER TABLE "media_assets"
ADD CONSTRAINT "media_assets_owner_id_fkey"
FOREIGN KEY ("owner_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_assets"
ADD CONSTRAINT "media_assets_pet_id_fkey"
FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_albums"
ADD CONSTRAINT "media_albums_owner_id_fkey"
FOREIGN KEY ("owner_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_albums"
ADD CONSTRAINT "media_albums_pet_id_fkey"
FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_album_assets"
ADD CONSTRAINT "media_album_assets_album_id_fkey"
FOREIGN KEY ("album_id") REFERENCES "media_albums"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_album_assets"
ADD CONSTRAINT "media_album_assets_asset_id_fkey"
FOREIGN KEY ("asset_id") REFERENCES "media_assets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_derivatives"
ADD CONSTRAINT "media_derivatives_asset_id_fkey"
FOREIGN KEY ("asset_id") REFERENCES "media_assets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_external_references"
ADD CONSTRAINT "media_external_references_asset_id_fkey"
FOREIGN KEY ("asset_id") REFERENCES "media_assets"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_external_references"
ADD CONSTRAINT "media_external_references_owner_id_fkey"
FOREIGN KEY ("owner_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "media_external_references"
ADD CONSTRAINT "media_external_references_pet_id_fkey"
FOREIGN KEY ("pet_id") REFERENCES "pets"("id") ON DELETE CASCADE ON UPDATE CASCADE;
