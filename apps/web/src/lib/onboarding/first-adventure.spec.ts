import { describe, expect, it } from 'vitest';
import {
  buildFirstAdventureResponses,
  emptyFirstAdventureSelections,
  type FirstAdventureSelections,
} from './first-adventure';

const completeSelections: FirstAdventureSelections = {
  goals: ['TRAINING', 'MORE_ADVENTURES', 'TRAINING'],
  timeBudget: 'TEN_TO_FIFTEEN',
  effort: 'MODERATE',
  socialComfort: 'SELECTIVELY_SOCIAL',
};

describe('First Adventure profile responses', () => {
  it('turns a sparse cold start into four explicit skips rather than blocking play', () => {
    const responses = buildFirstAdventureResponses('pet-1', emptyFirstAdventureSelections());

    expect(responses).toHaveLength(4);
    expect(responses.every((response) => response.outcome === 'SKIPPED')).toBe(true);
  });

  it('lets skip-all override already selected values without leaking answers', () => {
    const responses = buildFirstAdventureResponses('pet-1', completeSelections, true);

    expect(responses).toHaveLength(4);
    for (const response of responses) {
      expect(response.outcome).toBe('SKIPPED');
      expect(response).not.toHaveProperty('answers');
    }
  });

  it('records social uncertainty as NOT_SURE rather than a dislike', () => {
    const responses = buildFirstAdventureResponses('pet-1', {
      ...emptyFirstAdventureSelections(),
      socialComfort: 'NOT_SURE',
    });
    const social = responses.find(
      (response) => response.questionId === 'profile-dog-social-comfort-v1'
    );

    expect(social).toEqual({
      responseId: 'first-adventure-v1:pet-1:profile-dog-social-comfort-v1',
      questionId: 'profile-dog-social-comfort-v1',
      outcome: 'NOT_SURE',
    });
  });

  it('canonicalizes goals independently of click order or duplicate clicks', () => {
    const responses = buildFirstAdventureResponses('pet-1', completeSelections);
    const goals = responses.find(
      (response) => response.questionId === 'profile-owner-goals-v1'
    );

    expect(goals?.answers).toEqual(['MORE_ADVENTURES', 'TRAINING']);
  });

  it('uses stable replay identities for retries of the same pair and question', () => {
    const first = buildFirstAdventureResponses('pet-1', completeSelections);
    const replay = buildFirstAdventureResponses('pet-1', completeSelections);

    expect(replay.map((response) => response.responseId)).toEqual(
      first.map((response) => response.responseId)
    );
  });

  it('rejects more than three top goals before sending a payload', () => {
    expect(() =>
      buildFirstAdventureResponses('pet-1', {
        ...emptyFirstAdventureSelections(),
        goals: ['TRAINING', 'MORE_ADVENTURES', 'CALMER_ROUTINES', 'SOCIAL_CONFIDENCE'],
      })
    ).toThrow('at most three goals');
  });

  it('never includes game reward authority in personalization payloads', () => {
    const payload = JSON.stringify(buildFirstAdventureResponses('pet-1', completeSelections));

    expect(payload).not.toContain('bondXp');
    expect(payload).not.toContain('totalPoints');
    expect(payload).not.toContain('rewardLedger');
    expect(payload).not.toContain('pathwayXp');
  });
});
