import { WELLBEING_PATHWAYS } from '../care-events/care-event.types';
import { ADVENTURE_LEARNING_POLICY_VERSION } from './adventure-learning-policy';
import {
  buildAdventureLearningReceipt,
  parseCanonicalAdventureOutcome,
} from './adventure-learning-receipt';

describe('parseCanonicalAdventureOutcome', () => {
  it('accepts a canonical persisted outcome', () => {
    expect(
      parseCanonicalAdventureOutcome({
        dogExperience: 'loved_it',
        ownerExperience: 'a_lot_today',
        safeOptOut: true,
      })
    ).toEqual({
      dogExperience: 'loved_it',
      ownerExperience: 'a_lot_today',
      safeOptOut: true,
    });
  });

  it('treats an omitted legacy safe-opt-out flag as false', () => {
    expect(
      parseCanonicalAdventureOutcome({
        dogExperience: 'comfortable',
        ownerExperience: 'fine',
      })
    ).toEqual({
      dogExperience: 'comfortable',
      ownerExperience: 'fine',
      safeOptOut: false,
    });
  });

  it.each([
    null,
    undefined,
    {},
    { dogExperience: 'unknown', ownerExperience: 'fine' },
    { dogExperience: 'comfortable', ownerExperience: 'unknown' },
    { dogExperience: 'comfortable', ownerExperience: 'fine', safeOptOut: 'yes' },
  ])('fails closed for malformed persisted outcome %#', (value) => {
    expect(parseCanonicalAdventureOutcome(value as Record<string, unknown> | null | undefined)).toBe(
      null
    );
  });
});

describe('buildAdventureLearningReceipt', () => {
  it('uses the same policy version as the Adventure learning authority', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'LEARN',
      dogExperience: 'comfortable',
      ownerExperience: 'fine',
      safeOptOut: false,
    });

    expect(receipt.policyVersion).toBe(ADVENTURE_LEARNING_POLICY_VERSION);
  });

  it('treats a loved-it outcome as bounded positive fit evidence', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'EXPLORE',
      dogExperience: 'loved_it',
      ownerExperience: 'great',
      safeOptOut: false,
    });

    expect(receipt.headline).toBe('Worth remembering.');
    expect(receipt.dogSignal).toContain('positive fit signal for exploration');
    expect(receipt.nextRecommendationEffect).toContain('gently favored');
    expect(receipt.qualifier).toContain('not a permanent preference');
    expect(receipt.humanSignal).toBeNull();
  });

  it('keeps a comfortable outcome positive without claiming a permanent preference', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'BOND',
      dogExperience: 'comfortable',
      ownerExperience: 'fine',
      safeOptOut: false,
    });

    expect(receipt.headline).toBe('A useful comfortable data point.');
    expect(receipt.dogSignal).toContain('positive fit signal for bonding');
    expect(receipt.qualifier).toContain('One session is a clue');
  });

  it('treats not-their-thing as bounded negative fit evidence rather than failure', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'LEARN',
      dogExperience: 'not_their_thing',
      ownerExperience: 'fine',
      safeOptOut: false,
    });

    expect(receipt.headline).toBe('Useful discovery, not a failure.');
    expect(receipt.dogSignal).toContain('bounded negative fit signal for learning');
    expect(receipt.nextRecommendationEffect).toContain('gently deprioritized');
  });

  it('keeps owner load separate from dog preference and temporarily eases pace', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'EXPLORE',
      dogExperience: 'loved_it',
      ownerExperience: 'a_lot_today',
      safeOptOut: false,
    });

    expect(receipt.dogSignal).toContain('positive fit signal for exploration');
    expect(receipt.humanSignal).toContain("separate from your dog's preference");
    expect(receipt.nextRecommendationEffect).toContain('temporarily leans toward an easier pace');
  });

  it('can combine dog mismatch with temporary owner-load context without conflating them', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'CONNECT',
      dogExperience: 'not_their_thing',
      ownerExperience: 'a_lot_today',
      safeOptOut: false,
    });

    expect(receipt.dogSignal).toContain('bounded negative fit signal for social');
    expect(receipt.humanSignal).toContain("your load separate from your dog's preference");
    expect(receipt.nextRecommendationEffect).toContain('lean easier');
    expect(receipt.nextRecommendationEffect).toContain('less weight');
  });

  it('never turns a safe opt-out into durable dislike evidence', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'LEARN',
      dogExperience: 'not_their_thing',
      ownerExperience: 'fine',
      safeOptOut: true,
    });

    expect(receipt.headline).toBe('Good read.');
    expect(receipt.dogSignal).toContain('not counted as lasting evidence');
    expect(receipt.dogSignal).not.toContain('negative fit signal');
    expect(receipt.nextRecommendationEffect).toContain('lean gentler');
    expect(receipt.nextRecommendationEffect).toContain('recovery');
    expect(receipt.qualifier).toContain('successful outcome');
  });

  it('keeps high owner load separate even during a safe opt-out', () => {
    const receipt = buildAdventureLearningReceipt({
      pathway: 'CONNECT',
      dogExperience: 'comfortable',
      ownerExperience: 'a_lot_today',
      safeOptOut: true,
    });

    expect(receipt.humanSignal).toContain("separate from your dog's preference");
    expect(receipt.dogSignal).not.toContain('positive fit signal');
    expect(receipt.nextRecommendationEffect).toContain('next few days');
  });

  it.each(WELLBEING_PATHWAYS)('has user-safe pathway language for %s', (pathway) => {
    const receipt = buildAdventureLearningReceipt({
      pathway,
      dogExperience: 'comfortable',
      ownerExperience: 'fine',
      safeOptOut: false,
    });

    expect(receipt.dogSignal).not.toContain(pathway);
    expect(receipt.dogSignal.length).toBeGreaterThan(20);
    expect(receipt.nextRecommendationEffect.length).toBeGreaterThan(20);
  });

  it('is deterministic for the same canonical outcome', () => {
    const input = {
      pathway: 'RECOVER' as const,
      dogExperience: 'comfortable' as const,
      ownerExperience: 'a_lot_today' as const,
      safeOptOut: false,
    };

    expect(buildAdventureLearningReceipt(input)).toEqual(buildAdventureLearningReceipt(input));
  });
});
