import { PROFILE_QUESTION_POLICY_V1 } from './profile-question-policy-v1.receipt';
import type {
  ProfileDimension,
  ProfileEvidence,
  ProfileEvidenceProvenance,
  ProfileInteraction,
  ProfileQuestion,
  ProfileQuestionPolicyInput,
  ProfileQuestionTarget,
  QuestFamily,
  QuestionHistoryEntry,
} from './profile-question-policy-v1.types';

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;

const PROVENANCE_RANK: Record<ProfileEvidenceProvenance, number> = {
  OWNER_CORRECTION: 5,
  OWNER_EXPLICIT: 4,
  HOUSEHOLD_EXPLICIT: 3,
  HISTORICAL_QUIZ: 2,
  OUTCOME_INFERENCE: 1,
};

const PROFILE_STATE_RANK: Record<ProfileEvidence['state'], number> = {
  KNOWN: 3,
  LEARNING: 2,
  UNKNOWN: 1,
};

const QUESTION_OUTCOME_RANK: Record<QuestionHistoryEntry['outcome'], number> = {
  SKIPPED: 3,
  NOT_SURE: 2,
  ANSWERED: 1,
};

type Candidate = {
  id: string;
  target: ProfileQuestionTarget;
  prompt: string;
  whyAsk: string;
  answers: readonly string[];
  decisionValue: number;
  burden: number;
  safetyRelevance: number;
  contextTrigger: number;
  uncertainty: number;
  reasonCodes: string[];
};

type StaticQuestion = Omit<Candidate, 'uncertainty' | 'reasonCodes'> & {
  target: { kind: 'PROFILE'; dimension: ProfileDimension };
};

const STATIC_QUESTIONS: readonly StaticQuestion[] = [
  {
    id: 'profile-owner-goals-v1',
    target: { kind: 'PROFILE', dimension: 'OWNER_GOALS' },
    prompt: 'What would make Woof most useful for you and your dog right now?',
    whyAsk:
      'Goals help Woof choose between equally safe adventures instead of assuming everyone wants the same thing.',
    answers: [
      'MORE_ADVENTURES',
      'TRAINING',
      'CALMER_ROUTINES',
      'SOCIAL_CONFIDENCE',
      'CARE_ROUTINES',
      'JUST_HAVE_FUN',
      'NOT_SURE',
      'SKIP',
    ],
    decisionValue: 3,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 1,
  },
  {
    id: 'profile-owner-time-budget-v1',
    target: { kind: 'PROFILE', dimension: 'OWNER_TIME_BUDGET' },
    prompt: 'How much time usually feels realistic for one Woof adventure?',
    whyAsk:
      'A five-minute win can be a better recommendation than a thirty-minute plan that never fits the day.',
    answers: ['FIVE_MIN', 'TEN_TO_FIFTEEN', 'TWENTY_TO_THIRTY', 'FORTY_PLUS', 'VARIES', 'SKIP'],
    decisionValue: 3,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-owner-effort-v1',
    target: { kind: 'PROFILE', dimension: 'OWNER_EFFORT_PREFERENCE' },
    prompt: 'What effort level usually works best for you?',
    whyAsk:
      'Woof should fit the human half of the team too, rather than treating completion as the only outcome.',
    answers: ['KEEP_IT_EASY', 'MODERATE', 'UP_FOR_A_CHALLENGE', 'VARIES', 'SKIP'],
    decisionValue: 3,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-available-environments-v1',
    target: { kind: 'PROFILE', dimension: 'AVAILABLE_ENVIRONMENTS' },
    prompt: 'Where can adventures realistically happen most often?',
    whyAsk:
      'Environment changes which quests are possible and which training steps can generalize safely.',
    answers: ['INDOORS', 'YARD', 'NEIGHBORHOOD', 'PARK', 'TRAIL', 'VARIES', 'SKIP'],
    decisionValue: 2,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-dog-energy-v1',
    target: { kind: 'PROFILE', dimension: 'DOG_ENERGY_PATTERN' },
    prompt: 'Which broad energy pattern sounds most like your dog on an ordinary day?',
    whyAsk:
      'This is a starting prior only. Daily Signals and real outcomes can teach Woof when the pattern changes.',
    answers: ['MOSTLY_RESTFUL', 'MIXED', 'OFTEN_ACTIVE', 'HIGHLY_VARIABLE', 'NOT_SURE', 'SKIP'],
    decisionValue: 2,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-dog-social-comfort-v1',
    target: { kind: 'PROFILE', dimension: 'DOG_SOCIAL_COMFORT' },
    prompt: 'How does your dog usually feel about being near unfamiliar dogs?',
    whyAsk:
      'Social quests should start from known comfort and choice, not from an assumption that more interaction is better.',
    answers: [
      'PREFERS_SPACE',
      'CALM_AT_DISTANCE',
      'SELECTIVELY_SOCIAL',
      'OFTEN_SOCIAL',
      'NOT_SURE',
      'SKIP',
    ],
    decisionValue: 3,
    burden: 2,
    safetyRelevance: 1,
    contextTrigger: 0,
  },
  {
    id: 'profile-dog-novelty-v1',
    target: { kind: 'PROFILE', dimension: 'DOG_NOVELTY_COMFORT' },
    prompt: 'How does your dog usually respond to new places or unfamiliar activities?',
    whyAsk:
      'Novelty can be exciting for one dog and too much for another, so Woof should not use one exploration default.',
    answers: [
      'PREFERS_FAMILIAR',
      'WARMS_UP_SLOWLY',
      'USUALLY_CURIOUS',
      'HIGHLY_VARIABLE',
      'NOT_SURE',
      'SKIP',
    ],
    decisionValue: 2,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-dog-reinforcers-v1',
    target: { kind: 'PROFILE', dimension: 'DOG_REINFORCERS' },
    prompt: 'What tends to be most rewarding for your dog?',
    whyAsk:
      'Training is easier to personalize when Woof knows whether food, play, sniffing, praise, or another reinforcer actually matters to this dog.',
    answers: [
      'FOOD',
      'TOYS_PLAY',
      'SNIFFING_EXPLORING',
      'PRAISE_CONTACT',
      'VARIES',
      'NOT_SURE',
      'SKIP',
    ],
    decisionValue: 3,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
  {
    id: 'profile-dog-dislikes-v1',
    target: { kind: 'PROFILE', dimension: 'DOG_OBVIOUS_DISLIKES' },
    prompt: 'Are there activity types or situations your dog clearly prefers to avoid?',
    whyAsk:
      'Explicit dislikes should remove poor-fit suggestions instead of forcing the model to rediscover them through bad experiences.',
    answers: ['YES', 'NO_OBVIOUS_DISLIKES', 'NOT_SURE', 'SKIP'],
    decisionValue: 3,
    burden: 2,
    safetyRelevance: 1,
    contextTrigger: 0,
  },
  {
    id: 'profile-training-experience-v1',
    target: { kind: 'PROFILE', dimension: 'TRAINING_EXPERIENCE' },
    prompt: 'How much reward-based training have you and your dog done together?',
    whyAsk:
      'Training quests should start at a useful step for both halves of the team, not at a generic beginner or advanced level.',
    answers: [
      'NEW_TO_IT',
      'SOME_PRACTICE',
      'REGULAR_PRACTICE',
      'VERY_EXPERIENCED',
      'NOT_SURE',
      'SKIP',
    ],
    decisionValue: 2,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 0,
  },
];

function parseTimestamp(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function validProfileEvidence(
  evidence: readonly ProfileEvidence[],
  nowMs: number
): Map<ProfileDimension, ProfileEvidence> {
  const valid = evidence
    .map((entry) => ({ entry, updatedAtMs: parseTimestamp(entry.updatedAt) }))
    .filter(
      (item): item is { entry: ProfileEvidence; updatedAtMs: number } =>
        item.updatedAtMs !== null &&
        item.updatedAtMs <= nowMs &&
        Number.isFinite(item.entry.confidence) &&
        item.entry.confidence >= 0 &&
        item.entry.confidence <= 1
    )
    .sort((left, right) => {
      const dimension = left.entry.dimension.localeCompare(right.entry.dimension);
      if (dimension !== 0) return dimension;
      const provenance =
        PROVENANCE_RANK[right.entry.provenance] - PROVENANCE_RANK[left.entry.provenance];
      if (provenance !== 0) return provenance;
      const state = PROFILE_STATE_RANK[right.entry.state] - PROFILE_STATE_RANK[left.entry.state];
      if (state !== 0) return state;
      if (right.entry.confidence !== left.entry.confidence) {
        return right.entry.confidence - left.entry.confidence;
      }
      if (right.updatedAtMs !== left.updatedAtMs) return right.updatedAtMs - left.updatedAtMs;
      return left.entry.provenance.localeCompare(right.entry.provenance);
    });

  const byDimension = new Map<ProfileDimension, ProfileEvidence>();
  for (const item of valid) {
    if (!byDimension.has(item.entry.dimension)) byDimension.set(item.entry.dimension, item.entry);
  }
  return byDimension;
}

function interactionSignature(interaction: ProfileInteraction): string {
  return [
    interaction.occurredAt,
    interaction.kind,
    interaction.questFamily,
    interaction.dogExperience ?? '',
  ].join('|');
}

function recentInteractions(
  interactions: readonly ProfileInteraction[],
  nowMs: number
): Array<ProfileInteraction & { occurredAtMs: number }> {
  const windowStart = nowMs - PROFILE_QUESTION_POLICY_V1.interactionWindowDays * DAY_MS;
  const valid = interactions
    .map((interaction) => ({
      ...interaction,
      occurredAtMs: parseTimestamp(interaction.occurredAt),
    }))
    .filter(
      (interaction): interaction is ProfileInteraction & { occurredAtMs: number } =>
        interaction.occurredAtMs !== null &&
        interaction.occurredAtMs >= windowStart &&
        interaction.occurredAtMs <= nowMs
    )
    .sort((left, right) => {
      const id = left.id.localeCompare(right.id);
      if (id !== 0) return id;
      return interactionSignature(left).localeCompare(interactionSignature(right));
    });

  const byId = new Map<string, Array<ProfileInteraction & { occurredAtMs: number }>>();
  for (const interaction of valid) {
    const group = byId.get(interaction.id) ?? [];
    group.push(interaction);
    byId.set(interaction.id, group);
  }

  return [...byId.values()]
    .flatMap((group) => {
      const first = group[0];
      if (!first) return [];
      const signature = interactionSignature(first);
      if (group.some((interaction) => interactionSignature(interaction) !== signature)) return [];
      return [first];
    })
    .sort((left, right) => {
      if (left.occurredAtMs !== right.occurredAtMs) return left.occurredAtMs - right.occurredAtMs;
      return left.id.localeCompare(right.id);
    });
}

function inferredConfidence(
  dimension: ProfileDimension,
  interactions: readonly (ProfileInteraction & { occurredAtMs: number })[]
): number {
  const matching = interactions.filter((interaction) => {
    if (interaction.kind !== 'COMPLETED') return false;
    if (interaction.dogExperience !== 'loved_it' && interaction.dogExperience !== 'comfortable') {
      return false;
    }
    if (dimension === 'DOG_SOCIAL_COMFORT') return interaction.questFamily === 'SOCIAL';
    if (dimension === 'TRAINING_EXPERIENCE') return interaction.questFamily === 'TRAINING';
    return false;
  }).length;

  return Math.min(0.88, matching * 0.22);
}

function profileUncertainty(
  dimension: ProfileDimension,
  profile: Map<ProfileDimension, ProfileEvidence>,
  interactions: readonly (ProfileInteraction & { occurredAtMs: number })[]
): { uncertainty: number; known: boolean; reason: string } {
  const explicit = profile.get(dimension);
  const inferred = inferredConfidence(dimension, interactions);

  if (explicit?.provenance === 'OWNER_CORRECTION') {
    const known =
      explicit.state === 'KNOWN' &&
      explicit.confidence >= PROFILE_QUESTION_POLICY_V1.knownConfidenceThreshold;
    return {
      uncertainty: known ? 0 : 1 - explicit.confidence,
      known,
      reason: known ? 'OWNER_CORRECTION_KNOWN' : 'OWNER_CORRECTION_UNCERTAIN',
    };
  }

  const explicitConfidence = explicit?.state === 'UNKNOWN' ? 0 : (explicit?.confidence ?? 0);
  const effectiveConfidence = Math.max(explicitConfidence, inferred);
  const knownFromProfile =
    explicit?.state === 'KNOWN' &&
    explicit.confidence >= PROFILE_QUESTION_POLICY_V1.knownConfidenceThreshold;
  const knownFromOutcomes = inferred >= PROFILE_QUESTION_POLICY_V1.knownConfidenceThreshold;

  return {
    uncertainty: Math.max(0, 1 - effectiveConfidence),
    known: knownFromProfile || knownFromOutcomes,
    reason: knownFromProfile
      ? 'PROFILE_KNOWN'
      : knownFromOutcomes
        ? 'REPEATED_POSITIVE_OUTCOMES'
        : explicit
          ? 'PROFILE_LOW_CONFIDENCE'
          : 'PROFILE_UNKNOWN',
  };
}

function cooldownDays(entry: QuestionHistoryEntry): number {
  if (entry.outcome === 'SKIPPED') return PROFILE_QUESTION_POLICY_V1.skippedCooldownDays;
  if (entry.outcome === 'NOT_SURE') return PROFILE_QUESTION_POLICY_V1.notSureCooldownDays;
  return PROFILE_QUESTION_POLICY_V1.answeredCooldownDays;
}

function isInCooldown(
  questionId: string,
  history: readonly QuestionHistoryEntry[],
  nowMs: number
): boolean {
  const latest = history
    .filter((entry) => entry.questionId === questionId)
    .map((entry) => ({ entry, askedAtMs: parseTimestamp(entry.askedAt) }))
    .filter(
      (item): item is { entry: QuestionHistoryEntry; askedAtMs: number } =>
        item.askedAtMs !== null && item.askedAtMs <= nowMs
    )
    .sort((left, right) => {
      if (right.askedAtMs !== left.askedAtMs) return right.askedAtMs - left.askedAtMs;
      return QUESTION_OUTCOME_RANK[right.entry.outcome] - QUESTION_OUTCOME_RANK[left.entry.outcome];
    })[0];

  if (!latest) return false;
  return nowMs - latest.askedAtMs < cooldownDays(latest.entry) * DAY_MS;
}

function repeatedDismissalCandidates(
  interactions: readonly (ProfileInteraction & { occurredAtMs: number })[]
): Candidate[] {
  const counts = new Map<QuestFamily, number>();
  for (const interaction of interactions) {
    if (interaction.kind !== 'DISMISSED') continue;
    counts.set(interaction.questFamily, (counts.get(interaction.questFamily) ?? 0) + 1);
  }

  return [...counts.entries()]
    .filter(([, count]) => count >= PROFILE_QUESTION_POLICY_V1.dismissalsForClarification)
    .map(([questFamily, count]) => ({
      id: `profile-clarify-${questFamily.toLowerCase()}-preference-v1`,
      target: { kind: 'QUEST_FAMILY_PREFERENCE' as const, questFamily },
      prompt: `When ${questFamily.toLowerCase()} quests get passed over, is that usually preference or timing?`,
      whyAsk:
        'Woof should not turn a few declined suggestions into a permanent dislike when the day or timing may have been the real reason.',
      answers: ['NOT_A_FAVORITE', 'BAD_TIMING_LATELY', 'DEPENDS', 'NOT_SURE', 'SKIP'],
      decisionValue: 3,
      burden: 1,
      safetyRelevance: questFamily === 'SOCIAL' ? 1 : 0,
      contextTrigger: 3,
      uncertainty: 1,
      reasonCodes: ['REPEATED_RELATED_DISMISSALS', `DISMISSAL_COUNT_${count}`],
    }));
}

function trainingDifficultyCandidate(
  interactions: readonly (ProfileInteraction & { occurredAtMs: number })[],
  nowMs: number
): Candidate | null {
  const recentBoundary = nowMs - PROFILE_QUESTION_POLICY_V1.sessionFollowupWindowHours * HOUR_MS;
  const latest = [...interactions]
    .reverse()
    .find(
      (interaction) =>
        interaction.occurredAtMs >= recentBoundary &&
        interaction.kind === 'COMPLETED' &&
        interaction.questFamily === 'TRAINING' &&
        (interaction.dogExperience === 'loved_it' || interaction.dogExperience === 'comfortable')
    );

  if (!latest) return null;
  return {
    id: 'profile-training-session-difficulty-v1',
    target: { kind: 'SESSION_DIFFICULTY', questFamily: 'TRAINING' },
    prompt: 'How did that training step feel for the two of you?',
    whyAsk:
      'Difficulty feedback helps Woof distinguish a useful next step from an exercise that was technically completed but too easy or too demanding.',
    answers: ['EASY', 'ABOUT_RIGHT', 'CHALLENGING', 'NOT_SURE', 'SKIP'],
    decisionValue: 2,
    burden: 1,
    safetyRelevance: 0,
    contextTrigger: 3,
    uncertainty: 1,
    reasonCodes: ['RECENT_POSITIVE_TRAINING_COMPLETION'],
  };
}

function candidateScore(candidate: Candidate): number {
  const weights = PROFILE_QUESTION_POLICY_V1.scoreWeights;
  const raw =
    candidate.decisionValue * weights.decisionValue +
    candidate.uncertainty * weights.uncertainty +
    candidate.contextTrigger * weights.contextTrigger +
    candidate.safetyRelevance * weights.safetyRelevance +
    candidate.burden * weights.burden;
  return Math.round(raw * 1000) / 1000;
}

export function selectProfileQuestion(input: ProfileQuestionPolicyInput): ProfileQuestion | null {
  const nowMs = parseTimestamp(input.now);
  if (nowMs === null) {
    throw new Error('profile-question-policy-v1 requires a valid explicit now timestamp');
  }

  const profile = validProfileEvidence(input.profileEvidence, nowMs);
  const interactions = recentInteractions(input.interactions, nowMs);
  const candidates: Candidate[] = [];

  for (const question of STATIC_QUESTIONS) {
    const state = profileUncertainty(question.target.dimension, profile, interactions);
    if (state.known || isInCooldown(question.id, input.questionHistory, nowMs)) continue;

    const reasonCodes = [state.reason];
    if (question.contextTrigger > 0) reasonCodes.push('FOUNDATIONAL_COLD_START');
    if (question.safetyRelevance > 0) reasonCodes.push('ELIGIBILITY_RELEVANT_CONTEXT');
    candidates.push({ ...question, uncertainty: state.uncertainty, reasonCodes });
  }

  for (const candidate of repeatedDismissalCandidates(interactions)) {
    if (!isInCooldown(candidate.id, input.questionHistory, nowMs)) candidates.push(candidate);
  }

  const trainingCandidate = trainingDifficultyCandidate(interactions, nowMs);
  if (trainingCandidate && !isInCooldown(trainingCandidate.id, input.questionHistory, nowMs)) {
    candidates.push(trainingCandidate);
  }

  const ranked = candidates
    .map((candidate) => ({ candidate, score: candidateScore(candidate) }))
    .sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      return left.candidate.id.localeCompare(right.candidate.id);
    });

  const winner = ranked[0];
  if (!winner) return null;

  return {
    policyVersion: PROFILE_QUESTION_POLICY_V1.version,
    id: winner.candidate.id,
    target: winner.candidate.target,
    prompt: winner.candidate.prompt,
    whyAsk: winner.candidate.whyAsk,
    answers: winner.candidate.answers,
    score: winner.score,
    reasonCodes: winner.candidate.reasonCodes,
  };
}
