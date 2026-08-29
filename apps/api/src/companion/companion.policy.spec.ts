import { resolveCompanionState } from './companion.policy';

describe('Companion mode routing policy', () => {
  it('keeps a zero-pet account in explicit mode selection', () => {
    expect(resolveCompanionState(null, false)).toEqual({
      mode: null,
      modeSource: 'UNSET',
      hasAuthorizedPet: false,
      landing: 'NEEDS_MODE',
    });
  });

  it('preserves existing authorized-pet access without inventing a persisted identity mode', () => {
    expect(resolveCompanionState(null, true)).toEqual({
      mode: null,
      modeSource: 'PET_AUTHORITY_COMPAT',
      hasAuthorizedPet: true,
      landing: 'PET_TODAY',
    });
  });

  it('requires real pet authority before Pet Guardian reaches pet Today', () => {
    expect(resolveCompanionState('PET_GUARDIAN', false).landing).toBe('NEEDS_PET_SETUP');
    expect(resolveCompanionState('PET_GUARDIAN', true).landing).toBe('PET_TODAY');
  });

  it('lets Animal Ally and Foster Caregiver use Companion Today without a pet', () => {
    expect(resolveCompanionState('ANIMAL_ALLY', false).landing).toBe('COMPANION_TODAY');
    expect(resolveCompanionState('FOSTER_CAREGIVER', false).landing).toBe('COMPANION_TODAY');
  });

  it('does not let a presentation mode revoke existing pet relationship authority', () => {
    const state = resolveCompanionState('ANIMAL_ALLY', true);
    expect(state.hasAuthorizedPet).toBe(true);
    expect(state.landing).toBe('COMPANION_TODAY');
  });
});
