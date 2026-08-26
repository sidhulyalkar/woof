CREATE SCHEMA IF NOT EXISTS dogos_intelligence;

CREATE TABLE IF NOT EXISTS dogos_intelligence.observations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  pet_id TEXT NOT NULL REFERENCES public.pets(id) ON DELETE CASCADE,
  dimension TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_event_id TEXT,
  source_record_id TEXT,
  source_identity TEXT NOT NULL,
  observed_at TIMESTAMPTZ NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  local_date DATE NOT NULL,
  delta_bucket SMALLINT,
  numeric_value DOUBLE PRECISION,
  unit TEXT,
  confidence DOUBLE PRECISION NOT NULL,
  reliability TEXT NOT NULL,
  authority TEXT NOT NULL,
  normalization_version TEXT NOT NULL,
  normalization_reason TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  context JSONB NOT NULL DEFAULT '{}'::jsonb,
  supersedes_observation_id TEXT REFERENCES dogos_intelligence.observations(id) ON DELETE RESTRICT,
  retracted_at TIMESTAMPTZ,
  retraction_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT intelligence_observation_id_length
    CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT intelligence_observation_source_identity_length
    CHECK (char_length(source_identity) BETWEEN 1 AND 256),
  CONSTRAINT intelligence_observation_source_event_id_length
    CHECK (source_event_id IS NULL OR char_length(source_event_id) BETWEEN 1 AND 128),
  CONSTRAINT intelligence_observation_source_record_id_length
    CHECK (source_record_id IS NULL OR char_length(source_record_id) BETWEEN 1 AND 128),
  CONSTRAINT intelligence_observation_dimension
    CHECK (dimension IN (
      'APPETITE',
      'ENERGY',
      'BATHROOM_ROUTINE',
      'MOBILITY_COMFORT',
      'ENGAGEMENT_SOCIAL_COMFORT',
      'SLEEP_REST',
      'ACTIVITY_LOAD',
      'RECOVERY_REST_PROXY',
      'TRAINING_COMFORT_SUCCESS'
    )),
  CONSTRAINT intelligence_observation_source_type
    CHECK (source_type IN ('OWNER_CHECKIN', 'ACTIVITY', 'COACHING')),
  CONSTRAINT intelligence_observation_delta_bucket
    CHECK (delta_bucket IS NULL OR delta_bucket BETWEEN -2 AND 2),
  CONSTRAINT intelligence_observation_value_present
    CHECK (delta_bucket IS NOT NULL OR numeric_value IS NOT NULL),
  CONSTRAINT intelligence_observation_numeric_finite
    CHECK (
      numeric_value IS NULL
      OR numeric_value NOT IN ('Infinity'::float8, '-Infinity'::float8, 'NaN'::float8)
    ),
  CONSTRAINT intelligence_observation_unit_length
    CHECK (unit IS NULL OR char_length(unit) BETWEEN 1 AND 32),
  CONSTRAINT intelligence_observation_confidence
    CHECK (confidence > 0 AND confidence <= 1),
  CONSTRAINT intelligence_observation_reliability
    CHECK (reliability IN ('WEAK', 'STANDARD', 'STRONG')),
  CONSTRAINT intelligence_observation_authority
    CHECK (authority IN ('BASELINE_ELIGIBLE', 'CONTEXT_ONLY')),
  CONSTRAINT intelligence_observation_baseline_authority_shape
    CHECK (
      authority <> 'BASELINE_ELIGIBLE'
      OR (
        dimension IN (
          'APPETITE',
          'ENERGY',
          'BATHROOM_ROUTINE',
          'MOBILITY_COMFORT',
          'ENGAGEMENT_SOCIAL_COMFORT',
          'SLEEP_REST'
        )
        AND delta_bucket IS NOT NULL
      )
    ),
  CONSTRAINT intelligence_observation_normalization_version_length
    CHECK (char_length(normalization_version) BETWEEN 1 AND 64),
  CONSTRAINT intelligence_observation_reason_length
    CHECK (char_length(normalization_reason) BETWEEN 1 AND 512),
  CONSTRAINT intelligence_observation_payload_hash
    CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
  CONSTRAINT intelligence_observation_context_size
    CHECK (octet_length(context::text) <= 4096),
  CONSTRAINT intelligence_observation_retraction_reason_length
    CHECK (retraction_reason IS NULL OR char_length(retraction_reason) BETWEEN 1 AND 256),
  CONSTRAINT intelligence_observation_retraction_pair
    CHECK (
      (retracted_at IS NULL AND retraction_reason IS NULL)
      OR (retracted_at IS NOT NULL AND retraction_reason IS NOT NULL)
    ),
  CONSTRAINT intelligence_observation_not_self_superseding
    CHECK (supersedes_observation_id IS NULL OR supersedes_observation_id <> id),

  CONSTRAINT intelligence_observation_source_identity_unique
    UNIQUE (pet_id, dimension, source_type, source_identity, normalization_version)
);

CREATE INDEX IF NOT EXISTS intelligence_observations_pet_dimension_time_idx
  ON dogos_intelligence.observations (pet_id, dimension, observed_at DESC, id);

CREATE INDEX IF NOT EXISTS intelligence_observations_source_event_idx
  ON dogos_intelligence.observations (source_event_id, dimension)
  WHERE source_event_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS intelligence_observations_source_record_idx
  ON dogos_intelligence.observations (source_record_id, dimension)
  WHERE source_record_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS intelligence_observations_supersedes_idx
  ON dogos_intelligence.observations (supersedes_observation_id)
  WHERE supersedes_observation_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS intelligence_observations_one_active_superseder_idx
  ON dogos_intelligence.observations (supersedes_observation_id)
  WHERE supersedes_observation_id IS NOT NULL AND retracted_at IS NULL;
