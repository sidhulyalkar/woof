export const COMPANION_MODES = ['PET_GUARDIAN', 'ANIMAL_ALLY', 'FOSTER_CAREGIVER'] as const;

export type CompanionMode = (typeof COMPANION_MODES)[number];
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
        mode: 'PET_GUARDIAN',
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
