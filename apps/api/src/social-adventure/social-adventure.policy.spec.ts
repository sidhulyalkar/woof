import { currentUtcSeason, deriveSocialAdventureScore } from './social-adventure.policy';

describe('Social Adventure score policy', () => {
  it('rewards distinct Adventure variety rather than repetition volume', () => {
    const one = deriveSocialAdventureScore({
      adventurePathways: ['EXPLORE'],
      humanSkillBestScores: {},
    });
    const repeated = deriveSocialAdventureScore({
      adventurePathways: Array.from({ length: 100 }, () => 'EXPLORE'),
      humanSkillBestScores: {},
    });

    expect(one.score).toBe(25);
    expect(repeated.score).toBe(one.score);
    expect(repeated.components.adventureVariety.pathways).toEqual(['EXPLORE']);
  });

  it('does not turn CARE or unknown pathways into competitive points', () => {
    const score = deriveSocialAdventureScore({
      adventurePathways: ['CARE', 'MEDICAL', 'RECOVER'],
      humanSkillBestScores: {},
    });

    expect(score.components.adventureVariety.pathways).toEqual(['RECOVER']);
    expect(score.components.adventureVariety.score).toBe(25);
  });

  it('caps every Human Skill game and never scores posting or popularity inputs', () => {
    const score = deriveSocialAdventureScore({
      adventurePathways: [],
      humanSkillBestScores: {
        MAKE_IT_EASIER: 1000,
        CATCH_THE_GOOD: 88.4,
        PAIRING_LAB: -50,
        MARKER_TIMING: 92,
      },
    });

    expect(score.components.humanSkill.bestByChallenge).toEqual({
      MAKE_IT_EASIER: 100,
      CATCH_THE_GOOD: 88,
      PAIRING_LAB: 0,
      MARKER_TIMING: 92,
    });
    expect(score.components.humanSkill.score).toBe(280);
    expect(score.maxScore).toBe(575);
  });

  it('uses an explicit Monday UTC weekly season without streak semantics', () => {
    const season = currentUtcSeason(new Date('2026-08-27T21:00:00.000Z'));

    expect(season.key).toBe('week:2026-08-24');
    expect(season.startsAt.toISOString()).toBe('2026-08-24T00:00:00.000Z');
    expect(season.endsAt.toISOString()).toBe('2026-08-31T00:00:00.000Z');
  });
});
