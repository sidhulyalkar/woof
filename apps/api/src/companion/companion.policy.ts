export const COMPANION_MODES = ['PET_GUARDIAN', 'ANIMAL_ALLY', 'FOSTER_CAREGIVER'] as const;
export const READINESS_STATUSES = ['NOT_SURE', 'WORKING_ON_IT', 'READY_TO_DISCUSS'] as const;
export const READINESS_DIMENSIONS = [
  'housing',
  'householdAlignment',
  'timeCapacity',
  'financialPlan',
  'supportPlan',
  'carePlan',
] as const;

export type CompanionMode = (typeof COMPANION_MODES)[number];
export type ReadinessStatus = (typeof READINESS_STATUSES)[number];
export type ReadinessDimension = (typeof READINESS_DIMENSIONS)[number];
export type CompanionLanding =
  | 'NEEDS_MODE'
  | 'NEEDS_PET_SETUP'
  | 'PET_TODAY'
  | 'COMPANION_TODAY';

export type CompanionState = {
  mode: CompanionMode | null;
  modeSource: 'PERSISTED' | 'PET_AUTHORITY_COMPAT' | 'UNSET';
  hasAuthorizedPet: boolean;
  landing: CompanionLanding;
};

export function resolveCompanionState(
  mode: CompanionMode | null,
  hasAuthorizedPet: boolean
): CompanionState {
  if (!mode) {
    if (hasAuthorizedPet) {
      return {
        mode: null,
        modeSource: 'PET_AUTHORITY_COMPAT',
        hasAuthorizedPet,
        landing: 'PET_TODAY',
      };
    }
    return {
      mode: null,
      modeSource: 'UNSET',
      hasAuthorizedPet,
      landing: 'NEEDS_MODE',
    };
  }

  if (mode === 'PET_GUARDIAN') {
    return {
      mode,
      modeSource: 'PERSISTED',
      hasAuthorizedPet,
      landing: hasAuthorizedPet ? 'PET_TODAY' : 'NEEDS_PET_SETUP',
    };
  }

  return {
    mode,
    modeSource: 'PERSISTED',
    hasAuthorizedPet,
    landing: 'COMPANION_TODAY',
  };
}
