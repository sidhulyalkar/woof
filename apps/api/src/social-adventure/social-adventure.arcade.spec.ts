import { scenarioByKey, scoreArcadeResponse } from './social-adventure.arcade';

describe('Human Skill Arcade', () => {
  it('scores Make It Easier from the server-owned answer', () => {
    const scenario = scenarioByKey('recall-busy-park-v1');
    expect(scenario).not.toBeNull();
    if (!scenario) throw new Error('fixture missing');

    expect(scoreArcadeResponse(scenario, { optionId: 'quieter_context' })).toMatchObject({
      score: 100,
      correct: true,
    });
    expect(scoreArcadeResponse(scenario, { optionId: 'repeat_louder' })).toMatchObject({
      score: 20,
      correct: false,
    });
  });

  it('teaches pairing without requiring an operant behavior first', () => {
    const scenario = scenarioByKey('novel-sound-pairing-v1');
    expect(scenario).not.toBeNull();
    if (!scenario) throw new Error('fixture missing');

    const result = scoreArcadeResponse(scenario, { optionId: 'sound_then_good' });
    expect(result.score).toBe(100);
    expect(result.correct).toBe(true);
  });

  it('scores marker timing by temporal precision with bounded output', () => {
    const scenario = scenarioByKey('four-paws-mat-v1');
    expect(scenario).not.toBeNull();
    if (!scenario) throw new Error('fixture missing');

    expect(scoreArcadeResponse(scenario, { tapMs: 3200 })).toMatchObject({
      score: 100,
      correct: true,
      timingErrorMs: 0,
    });
    expect(scoreArcadeResponse(scenario, { tapMs: 4000 }).score).toBe(0);
    expect(scoreArcadeResponse(scenario, { tapMs: Number.NaN }).score).toBe(0);
  });
});
