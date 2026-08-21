import { DAILY_BOND_XP_CAP, rewardCareEvent } from './reward-policy';
import type { CareEventInput } from './care-event.types';

const baseEvent: CareEventInput = {
  userId: 'user-1',
  petId: 'pet-1',
  eventType: 'QUEST_EXPLORE',
  pathway: 'EXPLORE',
  source: 'QUEST_ENGINE',
  evidenceType: 'SELF_REPORT',
  evidenceConfidence: 0.65,
  dedupeKey: 'quest:one',
};

const emptyContext = {
  totalXpToday: 0,
  pathwayXpToday: 0,
  samePathwayEventsToday: 0,
  repeatedEventCount7d: 0,
};

describe('rewardCareEvent', () => {
  it('rewards recovery rather than treating rest as failure', () => {
    const decision = rewardCareEvent(
      { ...baseEvent, eventType: 'QUEST_RECOVER', pathway: 'RECOVER' },
      emptyContext,
    );

    expect(decision.bondXp).toBeGreaterThan(0);
    expect(decision.pathwayXp.RECOVER).toBe(decision.bondXp);
  });

  it('rewards a welfare-aware safe opt-out', () => {
    const decision = rewardCareEvent(
      {
        ...baseEvent,
        eventType: 'SAFE_OPT_OUT',
        pathway: 'BOND',
        outcome: { safeOptOut: true, dogExperience: 'not_their_thing' },
      },
      emptyContext,
    );

    expect(decision.bondXp).toBeGreaterThanOrEqual(18);
  });

  it('does not let optional media dominate the underlying action', () => {
    const withoutMemory = rewardCareEvent(baseEvent, emptyContext);
    const withMemory = rewardCareEvent(
      { ...baseEvent, context: { memoryAdded: true } },
      emptyContext,
    );

    expect(withMemory.bondXp - withoutMemory.bondXp).toBeLessThanOrEqual(2);
  });

  it('reduces farming value for repeated same-pathway events', () => {
    const first = rewardCareEvent(baseEvent, emptyContext);
    const repeated = rewardCareEvent(baseEvent, {
      ...emptyContext,
      samePathwayEventsToday: 4,
      repeatedEventCount7d: 8,
    });

    expect(repeated.bondXp).toBeLessThan(first.bondXp);
  });

  it('enforces the global daily cap', () => {
    const decision = rewardCareEvent(baseEvent, {
      ...emptyContext,
      totalXpToday: DAILY_BOND_XP_CAP,
    });

    expect(decision.bondXp).toBe(0);
  });

  it('fails closed for a safety-ineligible event', () => {
    const decision = rewardCareEvent(
      { ...baseEvent, safetyEligible: false },
      emptyContext,
    );

    expect(decision.bondXp).toBe(0);
    expect(decision.explanation).toContain('not safety-eligible');
  });
});
