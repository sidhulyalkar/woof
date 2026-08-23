CREATE SCHEMA IF NOT EXISTS dogos_discovery;

CREATE TABLE IF NOT EXISTS dogos_discovery.locations (
  user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  lat_bucket INTEGER NOT NULL,
  lng_bucket INTEGER NOT NULL,
  precision_m INTEGER NOT NULL DEFAULT 2200,
  enabled BOOLEAN NOT NULL DEFAULT TRUE,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT discovery_lat_bucket_range CHECK (lat_bucket BETWEEN 0 AND 9000),
  CONSTRAINT discovery_lng_bucket_range CHECK (lng_bucket BETWEEN 0 AND 18000),
  CONSTRAINT discovery_precision_range CHECK (precision_m BETWEEN 1000 AND 5000)
);

CREATE INDEX IF NOT EXISTS discovery_location_bucket_idx
  ON dogos_discovery.locations (lat_bucket, lng_bucket, expires_at)
  WHERE enabled = TRUE;
