import apiClient from './client';

export type CompanionMode = 'PET_GUARDIAN' | 'ANIMAL_ALLY' | 'FOSTER_CAREGIVER';
export type CompanionLanding = 'NEEDS_MODE' | 'NEEDS_PET_SETUP' | 'PET_TODAY' | 'COMPANION_TODAY';

export type CompanionState = {
  mode: CompanionMode | null;
  modeSource: 'PERSISTED' | 'PET_AUTHORITY_COMPAT' | 'UNSET';
  hasAuthorizedPet: boolean;
  landing: CompanionLanding;
  authority: {
    modeControlsPresentation: boolean;
    petAccessComesFromRelationships: boolean;
    modeNeverCreatesPetAuthority: boolean;
  };
};

export const companionApi = {
  state: () => apiClient.get<CompanionState>('/companion/state'),
  updateMode: (mode: CompanionMode) =>
    apiClient.put<CompanionState>('/companion/mode', { mode }),
};
