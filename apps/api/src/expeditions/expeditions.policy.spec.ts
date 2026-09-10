import {
  currentExpeditionSeason,
  EXPEDITION_ELIGIBLE_PATHWAYS,
  EXPEDITION_HUMAN_SKILL_CATEGORIES,
  EXPEDITION_OBJECTIVES,
} from './expeditions.policy';

describe('Expedition authority policy', () => {
  it('keeps CARE and medical pathways outside cooperative progress', () => {
    expect(EXPEDITION_ELIGIBLE_PATHWAYS).toEqual(['EXPLORE', 'ENRICH', 'RECOVER']);
    expect(EXPEDITION_ELIGIBLE_PATHWAYS).not.toContain('CARE' as never);
  });

  it('bounds every objective by contributor and category rather than volume', () => {
    for (const objective of EXPEDITION_OBJECTIVES) {
      expect(objective.perContributorCap).toBeGreaterThan(0);
      expect(objective.perCategoryCap).toBeGreaterThan(0);
      expect(objective.perCategoryCap * objective.categories.length).toBeLessThanOrEqual(
        objective.perContributorCap
      );
    }
  });

  it('counts Human Skill breadth once per distinct room without score magnitude', () => {
    const objective = EXPEDITION_OBJECTIVES.find((item) => item.key === 'READ_THE_ROOM');
    expect(objective?.sourceType).toBe('HUMAN_SKILL_ATTEMPT');
    expect(objective?.categories).toEqual([...EXPEDITION_HUMAN_SKILL_CATEGORIES]);
    expect(objective?.perCategoryCap).toBe(1);
    expect(objective?.perContributorCap).toBe(EXPEDITION_HUMAN_SKILL_CATEGORIES.length);

    // Protective copy may explicitly explain that score magnitude does not count.
    // Authority is structural: the objective definition itself carries no practice-
    // score, correctness, or timing field that could influence cooperative arithmetic.
    expect(objective).not.toHaveProperty('score');
    expect(objective).not.toHaveProperty('bestScore');
    expect(objective).not.toHaveProperty('practiceScore');
    expect(objective).not.toHaveProperty('correct');
    expect(objective).not.toHaveProperty('timingErrorMs');
    expect(objective).not.toHaveProperty('targetTimingMs');
  });

  it('uses an explicit Monday UTC season instead of a rolling streak window', () => {
    const season = currentExpeditionSeason(new Date('2026-09-09T18:00:00.000Z'));
    expect(season.key).toBe('week:2026-09-07');
    expect(season.startsAt.toISOString()).toBe('2026-09-07T00:00:00.000Z');
    expect(season.endsAt.toISOString()).toBe('2026-09-14T00:00:00.000Z');
  });
});
