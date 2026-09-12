-- Structured Pack locality authority.
--
-- IMPORTANT PRIVACY BOUNDARY:
-- Existing `region_key` values were arbitrary user-entered text and therefore
-- cannot be trusted as geographically coarse. Do not map, parse, log, or
-- reverse-geocode those values during migration. They are deliberately
-- discarded and legacy LOCAL packs remain available only through existing
-- membership until an owner selects an approved region through the API.

CREATE TABLE IF NOT EXISTS dogos_social.coarse_regions (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  country_code TEXT NOT NULL,
  subdivision_code TEXT,
  granularity TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT coarse_region_id_shape CHECK (id ~ '^[a-z0-9]+(?:-[a-z0-9]+)*$'),
  CONSTRAINT coarse_region_display_name_length CHECK (char_length(display_name) BETWEEN 2 AND 80),
  CONSTRAINT coarse_region_country_code CHECK (country_code ~ '^[A-Z]{2}$'),
  CONSTRAINT coarse_region_subdivision_code CHECK (
    subdivision_code IS NULL OR subdivision_code ~ '^[A-Z0-9-]{1,12}$'
  ),
  CONSTRAINT coarse_region_granularity CHECK (
    granularity IN ('METRO', 'COUNTY', 'BROAD_DISTRICT')
  )
);

-- v1 pilot catalog. These are broad public areas, not user coordinates.
-- Expansion is a reviewed catalog change, never a fallback to arbitrary text.
INSERT INTO dogos_social.coarse_regions
  (id, display_name, country_code, subdivision_code, granularity)
VALUES
  ('us-ca-san-francisco', 'San Francisco, CA', 'US', 'CA', 'METRO'),
  ('us-ca-south-bay', 'South Bay, CA', 'US', 'CA', 'BROAD_DISTRICT'),
  ('us-ca-peninsula', 'Peninsula, CA', 'US', 'CA', 'BROAD_DISTRICT'),
  ('us-ca-east-bay', 'East Bay, CA', 'US', 'CA', 'BROAD_DISTRICT'),
  ('us-ca-north-bay', 'North Bay, CA', 'US', 'CA', 'BROAD_DISTRICT'),
  ('us-ca-santa-cruz-county', 'Santa Cruz County, CA', 'US', 'CA', 'COUNTY')
ON CONFLICT (id) DO NOTHING;

DROP INDEX IF EXISTS dogos_social.social_packs_region_idx;

ALTER TABLE dogos_social.packs
  DROP CONSTRAINT IF EXISTS social_pack_region_shape;

-- Never infer an approved region from legacy user text. Purging the column
-- values is safer than preserving possibly precise addresses or venues under a
-- newly trusted field name. Memberships and pack identity remain intact.
UPDATE dogos_social.packs
SET region_key = NULL
WHERE scope = 'LOCAL';

ALTER TABLE dogos_social.packs
  ADD CONSTRAINT social_pack_region_authority_fk
    FOREIGN KEY (region_key)
    REFERENCES dogos_social.coarse_regions(id)
    ON UPDATE RESTRICT
    ON DELETE RESTRICT;

ALTER TABLE dogos_social.packs
  ADD CONSTRAINT social_pack_region_shape CHECK (
    (scope = 'LOCAL')
    OR (scope = 'FRIENDS' AND region_key IS NULL)
  );

CREATE INDEX social_packs_region_idx
  ON dogos_social.packs (region_key, created_at DESC)
  WHERE scope = 'LOCAL'
    AND visibility = 'PUBLIC'
    AND region_key IS NOT NULL;

COMMENT ON COLUMN dogos_social.packs.region_key IS
  'Server-approved coarse_regions.id. NULL on legacy LOCAL packs until owner repair; never arbitrary user location text.';
