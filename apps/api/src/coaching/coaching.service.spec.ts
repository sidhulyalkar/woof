import { recommendProgression, TrainingSignal } from './coaching.service';

const session = (successRate: number, options: Partial<TrainingSignal> = {}): TrainingSignal => ({
  attempts: 5,
  successes: Math.round(successRate * 5),
  successRate,
  stressSignals: [],
  stoppedEarly: false,
  difficultyLevel: 2,
  distractionLevel: 1,
  ...options,
});

describe('reward-based coaching progression', () => {
  it('starts at an easy level when there is no history', () => {
    const decision = recommendProgression([], 1);
    expect(decision.action).toBe('start');
    expect(decision.nextLevel).toBe(1);
  });

  it('adds only one level after two comfortable successful sessions', () => {
    const decision = recommendProgression([session(1), session(0.8)], 2);
    expect(decision.action).toBe('increase');
    expect(decision.nextLevel).toBe(3);
  });

  it('holds when evidence is not yet strong enough to progress', () => {
    const decision = recommendProgression([session(0.8)], 2);
    expect(decision.action).toBe('hold');
    expect(decision.nextLevel).toBe(2);
  });

  it('reduces difficulty instead of repeating a failing setup', () => {
    const decision = recommendProgression(
      [session(0.4, { difficultyLevel: 3 }), session(0.4, { difficultyLevel: 3 })],
      3
    );
    expect(decision.action).toBe('decrease');
    expect(decision.nextLevel).toBe(2);
  });

  it('reduces difficulty immediately when the pet stops or shows a strong concern signal', () => {
    const decision = recommendProgression(
      [session(0.8, { stoppedEarly: true, stressSignals: ['escape-attempt'] })],
      3
    );
    expect(decision.action).toBe('decrease');
    expect(decision.nextLevel).toBe(2);
    expect(decision.reason).toContain('distance');
  });

  it('never increments past the maximum difficulty', () => {
    const decision = recommendProgression([session(1), session(1)], 5);
    expect(decision.action).toBe('hold');
    expect(decision.nextLevel).toBe(5);
  });
});

describe('comparable coaching evidence', () => {
  it('does not reuse an easier level to advance after one new-level session', () => {
    expect(recommendProgression([session(1), session(1, { difficultyLevel: 1 })], 2).action).toBe(
      'hold'
    );
  });
  it('does not combine successful sessions from different distraction contexts', () => {
    expect(recommendProgression([session(1), session(1, { distractionLevel: 3 })], 2).action).toBe(
      'hold'
    );
  });
  it('keeps unknown legacy context from authorizing progression', () => {
    expect(
      recommendProgression([session(1, { distractionLevel: undefined }), session(1)], 2).action
    ).toBe('hold');
  });
  it('can recommend stopping before any repetition', () => {
    expect(
      recommendProgression([session(0, { attempts: 0, successes: 0, stoppedEarly: true })], 2)
        .nextLevel
    ).toBe(1);
  });
});
