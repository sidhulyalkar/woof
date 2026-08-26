import {
  deriveAdventureLearningSignals,
  type AdventureLearningEvent,
} from './adventure-learning-policy';

const NOW = new Date('2026-08-26T12:00:00.000Z');

function event(
  overrides: Partial<AdventureLearningEvent> & Pick<AdventureLearningEvent, 'id'>
): AdventureLearningEvent {
  return {
    id: overrides.id,
    eventType: overrides.eventType ?? 'QUEST_BOND',
    pathway: overrides.pathway ?? 'BOND',
    occurredAt: overrides.occurredAt ?? '2026-08-26T10:00:00.000Z',
    context: overrides.context ?? {},
    outcome: overrides.outcome ?? {},
  };
}

describe('Adventure learning v2', () => {
  it('keeps a safe LEARN opt-out out of durable BOND and LEARN preference', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'safe-stop',
          pathway: 'BOND',
          context: { originalPathway: 'LEARN' },
          outcome: {
            dogExperience: 'not_their_thing',
            ownerExperience: 'fine',
            safeOptOut: true,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.BOND).toBeUndefined();
    expect(result.durablePathwayPreference.LEARN).toBeUndefined();
    expect(result.temporaryPathwayModifier.LEARN).toBeLessThan(0);
    expect(result.temporaryPathwayModifier.RECOVER).toBeGreaterThan(0);
    expect(result.temporaryPace).toBe('easy');
  });

  it('targets non-safe dog mismatch at the original pathway, never the BOND reward pathway', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'mismatch',
          pathway: 'BOND',
          context: { originalPathway: 'CONNECT' },
          outcome: {
            dogExperience: 'not_their_thing',
            ownerExperience: 'fine',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.CONNECT).toBeLessThan(0);
    expect(result.durablePathwayPreference.BOND).toBeUndefined();
  });

  it('treats owner overload as temporary context instead of durable dog preference', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'busy-owner',
          pathway: 'MOVE',
          context: { originalPathway: 'MOVE' },
          outcome: {
            dogExperience: 'loved_it',
            ownerExperience: 'a_lot_today',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.MOVE).toBeUndefined();
    expect(result.temporaryPathwayModifier.MOVE).toBeLessThan(0);
    expect(result.temporaryPathwayModifier.RECOVER).toBeGreaterThan(0);
    expect(result.temporaryPace).toBe('easy');
  });

  it('expires temporary owner-load context without erasing durable history', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'old-owner-load',
          pathway: 'MOVE',
          occurredAt: '2026-08-22T12:00:00.000Z',
          context: { originalPathway: 'MOVE' },
          outcome: {
            dogExperience: 'comfortable',
            ownerExperience: 'a_lot_today',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.MOVE).toBeUndefined();
    expect(result.temporaryPathwayModifier).toEqual({});
    expect(result.temporaryPace).toBe('normal');
  });

  it('accumulates repeated positive dog outcomes monotonically while remaining bounded', () => {
    const one = deriveAdventureLearningSignals(
      [
        event({
          id: 'positive-1',
          pathway: 'LEARN',
          context: { originalPathway: 'LEARN' },
          outcome: { dogExperience: 'loved_it', ownerExperience: 'great', safeOptOut: false },
        }),
      ],
      NOW
    );
    const many = deriveAdventureLearningSignals(
      Array.from({ length: 12 }, (_, index) =>
        event({
          id: `positive-${index}`,
          pathway: 'LEARN',
          context: { originalPathway: 'LEARN' },
          outcome: { dogExperience: 'loved_it', ownerExperience: 'great', safeOptOut: false },
        })
      ),
      NOW
    );

    expect(one.durablePathwayPreference.LEARN).toBeGreaterThan(0);
    expect(many.durablePathwayPreference.LEARN).toBeGreaterThan(
      one.durablePathwayPreference.LEARN ?? 0
    );
    expect(many.durablePathwayPreference.LEARN).toBeLessThanOrEqual(0.08);
  });

  it('does not let one mismatch erase several recent positive outcomes', () => {
    const result = deriveAdventureLearningSignals(
      [
        ...Array.from({ length: 3 }, (_, index) =>
          event({
            id: `liked-${index}`,
            pathway: 'LEARN',
            context: { originalPathway: 'LEARN' },
            outcome: { dogExperience: 'loved_it', ownerExperience: 'fine', safeOptOut: false },
          })
        ),
        event({
          id: 'one-mismatch',
          pathway: 'BOND',
          context: { originalPathway: 'LEARN' },
          outcome: {
            dogExperience: 'not_their_thing',
            ownerExperience: 'fine',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.LEARN).toBeGreaterThan(0);
  });

  it('ignores ambiguous legacy BOND mismatch rather than corrupting Bond preference', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'legacy-mismatch',
          pathway: 'BOND',
          context: {},
          outcome: {
            dogExperience: 'not_their_thing',
            ownerExperience: 'fine',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference).toEqual({});
  });

  it('falls back to the event pathway for unambiguous legacy positive outcomes', () => {
    const result = deriveAdventureLearningSignals(
      [
        event({
          id: 'legacy-positive',
          pathway: 'EXPLORE',
          context: {},
          outcome: {
            dogExperience: 'comfortable',
            ownerExperience: 'fine',
            safeOptOut: false,
          },
        }),
      ],
      NOW
    );

    expect(result.durablePathwayPreference.EXPLORE).toBeGreaterThan(0);
  });

  it('is deterministic under input permutation', () => {
    const events = [
      event({
        id: 'a',
        pathway: 'MOVE',
        occurredAt: '2026-08-26T08:00:00.000Z',
        context: { originalPathway: 'MOVE' },
        outcome: { dogExperience: 'loved_it', ownerExperience: 'fine', safeOptOut: false },
      }),
      event({
        id: 'b',
        pathway: 'BOND',
        occurredAt: '2026-08-25T08:00:00.000Z',
        context: { originalPathway: 'LEARN' },
        outcome: { dogExperience: 'not_their_thing', ownerExperience: 'fine', safeOptOut: true },
      }),
    ];

    expect(deriveAdventureLearningSignals(events, NOW)).toEqual(
      deriveAdventureLearningSignals([...events].reverse(), NOW)
    );
  });

  it('does not use reward XP as a learning input', () => {
    const lowReward = event({
      id: 'xp-invariant',
      pathway: 'LEARN',
      context: { originalPathway: 'LEARN' },
      outcome: { dogExperience: 'loved_it', ownerExperience: 'great', safeOptOut: false },
    });
    const highReward = { ...lowReward, bondXp: 10_000 } as AdventureLearningEvent & {
      bondXp: number;
    };

    expect(deriveAdventureLearningSignals([lowReward], NOW)).toEqual(
      deriveAdventureLearningSignals([highReward], NOW)
    );
  });
});
