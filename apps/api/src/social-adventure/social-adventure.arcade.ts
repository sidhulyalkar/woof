import {
  HUMAN_SKILL_CHALLENGE_VERSION,
  type HumanSkillChallenge,
} from './social-adventure.policy';

export type ArcadeOption = {
  id: string;
  label: string;
};

export type PublicArcadeScenario = {
  challengeKey: HumanSkillChallenge;
  challengeVersion: string;
  scenarioKey: string;
  title: string;
  skill: string;
  prompt: string;
  options?: ArcadeOption[];
  timing?: {
    durationMs: number;
    targetAtMs: number;
    targetLabel: string;
  };
};

type ChoiceScenario = PublicArcadeScenario & {
  correctOptionId: string;
  explanation: string;
  options: ArcadeOption[];
};

type TimingScenario = PublicArcadeScenario & {
  timing: NonNullable<PublicArcadeScenario['timing']>;
  explanation: string;
};

type ArcadeScenario = ChoiceScenario | TimingScenario;

const SCENARIOS: ArcadeScenario[] = [
  {
    challengeKey: 'MAKE_IT_EASIER',
    challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
    scenarioKey: 'recall-busy-park-v1',
    title: 'Make It Easier',
    skill: 'Difficulty selection',
    prompt:
      'A recall is easy at home but falls apart in a busy park. What is the best next training move?',
    options: [
      { id: 'repeat_louder', label: 'Repeat the cue more loudly until it works' },
      { id: 'quieter_context', label: 'Move to a quieter context and rebuild success there' },
      { id: 'remove_rewards', label: 'Remove rewards so the cue becomes less dependent on food' },
      { id: 'add_distance', label: 'Increase distance so the dog learns to focus harder' },
    ],
    correctOptionId: 'quieter_context',
    explanation:
      'Change one difficulty variable at a time. Lowering distraction can restore a learnable setup without treating one hard context as stubbornness.',
  },
  {
    challengeKey: 'CATCH_THE_GOOD',
    challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
    scenarioKey: 'quiet-settle-v1',
    title: 'Catch the Good',
    skill: 'Reinforcing useful alternatives',
    prompt:
      'Your dog has been pacing while you make dinner, then quietly lies on a mat without being asked. What is the most useful moment to reinforce?',
    options: [
      { id: 'wait_for_pacing', label: 'Wait for pacing to restart so you can interrupt it' },
      { id: 'mark_settle', label: 'Mark and reinforce the voluntary settle on the mat' },
      { id: 'call_off_mat', label: 'Call the dog off the mat and ask for a formal sit' },
      { id: 'nothing', label: 'Do nothing because you did not cue the behavior' },
    ],
    correctOptionId: 'mark_settle',
    explanation:
      'Useful behavior does not have to be prompted first. Reinforcing the behavior you want to see more often can be more informative than waiting for an error.',
  },
  {
    challengeKey: 'PAIRING_LAB',
    challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
    scenarioKey: 'novel-sound-pairing-v1',
    title: 'Pairing Lab',
    skill: 'Positive association timing',
    prompt:
      'A mild novel sound is comfortable enough that the dog can notice it without distress. Which pairing best teaches that the sound predicts something good?',
    options: [
      { id: 'sound_then_good', label: 'Sound appears, then the good thing follows promptly' },
      { id: 'good_then_sound_late', label: 'Give the good thing, wait a long time, then play the sound' },
      { id: 'require_sit', label: 'Play the sound, require a sit, then reward the sit' },
      { id: 'repeat_until_ignored', label: 'Repeat the sound continuously until it is ignored' },
    ],
    correctOptionId: 'sound_then_good',
    explanation:
      'For a positive association, the stimulus should reliably predict the positive event. This is not a license to push exposure intensity; significant fear or aggression belongs with qualified professional support.',
  },
  {
    challengeKey: 'MARKER_TIMING',
    challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
    scenarioKey: 'four-paws-mat-v1',
    title: 'Marker Timing',
    skill: 'Temporal precision',
    prompt:
      'Watch the short timing track and mark the instant the target behavior occurs: all four paws arrive on the mat.',
    timing: {
      durationMs: 5200,
      targetAtMs: 3200,
      targetLabel: 'Four paws on mat',
    },
    explanation:
      'The marker is most useful when it identifies the behavior precisely. In real training, clarity matters more than chasing a game-perfect millisecond score.',
  },
];

export function publicArcadeCatalog() {
  return SCENARIOS.map((scenario) => toPublicScenario(scenario));
}

export function scenarioForChallenge(challengeKey: HumanSkillChallenge) {
  return SCENARIOS.find((scenario) => scenario.challengeKey === challengeKey) ?? null;
}

export function scenarioByKey(scenarioKey: string) {
  return SCENARIOS.find((scenario) => scenario.scenarioKey === scenarioKey) ?? null;
}

export function scoreArcadeResponse(
  scenario: ArcadeScenario,
  response: Record<string, unknown>
): { score: number; correct: boolean; explanation: string; timingErrorMs?: number } {
  if ('correctOptionId' in scenario) {
    const answer = typeof response.optionId === 'string' ? response.optionId : '';
    const correct = answer === scenario.correctOptionId;
    return {
      score: correct ? 100 : 20,
      correct,
      explanation: scenario.explanation,
    };
  }

  const tapMs = typeof response.tapMs === 'number' && Number.isFinite(response.tapMs)
    ? response.tapMs
    : Number.NaN;
  if (!Number.isFinite(tapMs) || tapMs < 0 || tapMs > scenario.timing.durationMs) {
    return {
      score: 0,
      correct: false,
      explanation: 'The timing response was outside the playable window.',
    };
  }

  const timingErrorMs = Math.round(Math.abs(tapMs - scenario.timing.targetAtMs));
  const score = Math.max(0, Math.min(100, Math.round(100 - timingErrorMs / 8)));
  return {
    score,
    correct: timingErrorMs <= 120,
    timingErrorMs,
    explanation: scenario.explanation,
  };
}

function toPublicScenario(scenario: ArcadeScenario): PublicArcadeScenario {
  return {
    challengeKey: scenario.challengeKey,
    challengeVersion: scenario.challengeVersion,
    scenarioKey: scenario.scenarioKey,
    title: scenario.title,
    skill: scenario.skill,
    prompt: scenario.prompt,
    ...(scenario.options ? { options: scenario.options } : {}),
    ...(scenario.timing ? { timing: scenario.timing } : {}),
  };
}
