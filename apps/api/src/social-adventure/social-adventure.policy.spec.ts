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

  it('rewards bounded Human Skill breadth while ignoring practice score magnitude', () => {
    const lowPractice = deriveSocialAdventureScore({
      adventurePathways: [],
      humanSkillBestScores: {
        MAKE_IT_EASIER: 0,
        CATCH_THE_GOOD: 1,
        PAIRING_LAB: -50,
        MARKER_TIMING: 2,
      },
    });
    const perfectPractice = deriveSocialAdventureScore({
      adventurePathways: [],
      humanSkillBestScores: {
        MAKE_IT_EASIER: 100,
        CATCH_THE_GOOD: 100,
        PAIRING_LAB: 100,
        MARKER_TIMING: 100,
      },
    });

    expect(lowPractice.components.humanSkill.completedChallenges).toEqual([
      'MAKE_IT_EASIER',
      'CATCH_THE_GOOD',
      'PAIRING_LAB',
      'MARKER_TIMING',
    ]);
    expect(lowPractice.components.humanSkill.score).toBe(200);
    expect(perfectPractice.score).toBe(lowPractice.score);
    expect(lowPractice.maxScore).toBe(375);
  });

  it('ignores missing or non-finite practice telemetry', () => {
    const score = deriveSocialAdventureScore({
      adventurePathways: [],
      humanSkillBestScores: {
        MAKE_IT_EASIER: Number.NaN,
        MARKER_TIMING: Number.POSITIVE_INFINITY,
      },
    });

    expect(score.components.humanSkill.completedChallenges).toEqual([]);
    expect(score.components.humanSkill.score).toBe(0);
  });

  it('uses an explicit Monday UTC weekly season without streak semantics', () => {
    const season = currentUtcSeason(new Date('2026-08-27T21:00:00.000Z'));

    expect(season.key).toBe('week:2026-08-24');
    expect(season.startsAt.toISOString()).toBe('2026-08-24T00:00:00.000Z');
    expect(season.endsAt.toISOString()).toBe('2026-08-31T00:00:00.000Z');
  });
});
