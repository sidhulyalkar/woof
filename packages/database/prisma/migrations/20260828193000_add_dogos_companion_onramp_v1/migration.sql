CREATE SCHEMA IF NOT EXISTS dogos_companion;

CREATE TABLE dogos_companion.profiles (
  user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  mode TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT companion_profile_mode_valid
    CHECK (mode IS NULL OR mode IN ('PET_GUARDIAN', 'ANIMAL_ALLY', 'FOSTER_CAREGIVER'))
);

CREATE INDEX companion_profiles_mode_idx
  ON dogos_companion.profiles (mode)
  WHERE mode IS NOT NULL;

CREATE FUNCTION dogos_companion.readiness_dimensions_valid(value JSONB)
RETURNS BOOLEAN
LANGUAGE SQL
IMMUTABLE
AS $$
  SELECT
    jsonb_typeof(value) = 'object'
    AND NOT EXISTS (
      SELECT 1
      FROM jsonb_each_text(value) AS entry(key, status)
      WHERE key NOT IN (
        'housing',
        'householdAlignment',
        'timeCapacity',
        'financialPlan',
        'supportPlan',
        'carePlan'
      )
      OR status NOT IN ('NOT_SURE', 'WORKING_ON_IT', 'READY_TO_DISCUSS')
    );
$$;

CREATE TABLE dogos_companion.readiness_reflections (
  user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT readiness_dimensions_bounded
    CHECK (dogos_companion.readiness_dimensions_valid(dimensions))
);

-- Preserve the existing guardian experience for users who already own a pet.
-- Shared household access intentionally does not backfill a global identity mode:
-- caregiver authority belongs to the relationship, not the account label.
INSERT INTO dogos_companion.profiles (user_id, mode)
SELECT DISTINCT pet.owner_id, 'PET_GUARDIAN'
FROM public.pets pet
JOIN public.users account ON account.id = pet.owner_id
ON CONFLICT (user_id) DO NOTHING;
