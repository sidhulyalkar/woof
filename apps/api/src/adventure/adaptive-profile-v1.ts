import { PROFILE_QUESTION_POLICY_V1 } from './profile-question-policy-v1.receipt';
import {
  PROFILE_DIMENSIONS,
  type ProfileDimension,
  type ProfileEvidence,
  type ProfileEvidenceProvenance,
  type QuestionHistoryOutcome,
} from './profile-question-policy-v1.types';

export const ADAPTIVE_PROFILE_SCHEMA_VERSION = 'adaptive-profile-v1' as const;

export type AdaptiveProfileSubject = 'DOG' | 'OWNER' | 'PAIR';

export type PersistedProfileEvidenceLike = {
  id: string;
  dimension: string;
  subject: string;
  state: string;
  value: unknown;
  confidence: number;
  provenance: string;
  schemaVersion: string;
  occurredAt: Date;
};

type StaticProfileQuestionDefinition = {
  dimension: ProfileDimension;
  answers: readonly string[];
  maxSelections: number;
};

const QUESTION_DEFINITIONS: Readonly<Record<string, StaticProfileQuestionDefinition>> = {
  'profile-owner-goals-v1': {
    dimension: 'OWNER_GOALS',
    answers: [
      'MORE_ADVENTURES',
      'TRAINING',
      'CALMER_ROUTINES',
      'SOCIAL_CONFIDENCE',
      'CARE_ROUTINES',
      'JUST_HAVE_FUN',
    ],
    maxSelections: 3,
  },
  'profile-owner-time-budget-v1': {
    dimension: 'OWNER_TIME_BUDGET',
    answers: ['FIVE_MIN', 'TEN_TO_FIFTEEN', 'TWENTY_TO_THIRTY', 'FORTY_PLUS', 'VARIES'],
    maxSelections: 1,
  },
  'profile-owner-effort-v1': {
    dimension: 'OWNER_EFFORT_PREFERENCE',
    answers: ['KEEP_IT_EASY', 'MODERATE', 'UP_FOR_A_CHALLENGE', 'VARIES'],
    maxSelections: 1,
  },
  'profile-available-environments-v1': {
    dimension: 'AVAILABLE_ENVIRONMENTS',
    answers: ['INDOORS', 'YARD', 'NEIGHBORHOOD', 'PARK', 'TRAIL', 'VARIES'],
    maxSelections: 6,
  },
  'profile-dog-energy-v1': {
    dimension: 'DOG_ENERGY_PATTERN',
    answers: ['MOSTLY_RESTFUL', 'MIXED', 'OFTEN_ACTIVE', 'HIGHLY_VARIABLE'],
    maxSelections: 1,
  },
  'profile-dog-social-comfort-v1': {
    dimension: 'DOG_SOCIAL_COMFORT',
    answers: ['PREFERS_SPACE', 'CALM_AT_DISTANCE', 'SELECTIVELY_SOCIAL', 'OFTEN_SOCIAL'],
    maxSelections: 1,
  },
  'profile-dog-novelty-v1': {
    dimension: 'DOG_NOVELTY_COMFORT',
    answers: ['PREFERS_FAMILIAR', 'WARMS_UP_SLOWLY', 'USUALLY_CURIOUS', 'HIGHLY_VARIABLE'],
    maxSelections: 1,
  },
  'profile-dog-reinforcers-v1': {
    dimension: 'DOG_REINFORCERS',
    answers: ['FOOD', 'TOYS_PLAY', 'SNIFFING_EXPLORING', 'PRAISE_CONTACT', 'VARIES'],
    maxSelections: 5,
  },
  'profile-dog-dislikes-v1': {
    dimension: 'DOG_OBVIOUS_DISLIKES',
    answers: ['YES', 'NO_OBVIOUS_DISLIKES'],
    maxSelections: 1,
  },
  'profile-training-experience-v1': {
    dimension: 'TRAINING_EXPERIENCE',
    answers: ['NEW_TO_IT', 'SOME_PRACTICE', 'REGULAR_PRACTICE', 'VERY_EXPERIENCED'],
    maxSelections: 1,
  },
};

const SUBJECT_BY_DIMENSION: Record<ProfileDimension, AdaptiveProfileSubject> = {
  OWNER_GOALS: 'OWNER',
  OWNER_TIME_BUDGET: 'OWNER',
  OWNER_EFFORT_PREFERENCE: 'OWNER',
  AVAILABLE_ENVIRONMENTS: 'PAIR',
  DOG_ENERGY_PATTERN: 'DOG',
  DOG_SOCIAL_COMFORT: 'DOG',
  DOG_NOVELTY_COMFORT: 'DOG',
  DOG_REINFORCERS: 'DOG',
  DOG_OBVIOUS_DISLIKES: 'DOG',
  TRAINING_EXPERIENCE: 'PAIR',
};

const PROVENANCE_RANK: Record<ProfileEvidenceProvenance, number> = {
  OWNER_CORRECTION: 5,
  OWNER_EXPLICIT: 4,
  HOUSEHOLD_EXPLICIT: 3,
  HISTORICAL_QUIZ: 2,
  OUTCOME_INFERENCE: 1,
};

const STATE_RANK: Record<ProfileEvidence['state'], number> = {
  KNOWN: 3,
  LEARNING: 2,
  UNKNOWN: 1,
};

export function isProfileDimension(value: string): value is ProfileDimension {
  return (PROFILE_DIMENSIONS as readonly string[]).includes(value);
}

export function profileSubjectForDimension(dimension: ProfileDimension): AdaptiveProfileSubject {
  return SUBJECT_BY_DIMENSION[dimension];
}

export function staticProfileQuestion(questionId: string): StaticProfileQuestionDefinition | null {
  return QUESTION_DEFINITIONS[questionId] ?? null;
}

export function normalizeQuestionAnswer(
  questionId: string,
  outcome: QuestionHistoryOutcome,
  answers: readonly string[] | undefined
): { dimension: ProfileDimension; values: string[] | null } | null {
  const definition = staticProfileQuestion(questionId);
  if (!definition) return null;

  if (outcome !== 'ANSWERED') {
    if (answers?.length) throw new Error('Non-answered profile questions cannot carry answers');
    return { dimension: definition.dimension, values: null };
  }

  const values = [...new Set(answers ?? [])].sort();
  if (!values.length || values.length > definition.maxSelections) {
    throw new Error('Profile answer count is outside the allowed range');
  }
  if (values.some((value) => !definition.answers.includes(value))) {
    throw new Error('Profile answer is not valid for this question');
  }
  return { dimension: definition.dimension, values };
}

export function resolveCurrentProfileEvidence(
  evidence: readonly PersistedProfileEvidenceLike[]
): PersistedProfileEvidenceLike[] {
  const valid = evidence.filter(
    (entry) =>
      entry.schemaVersion === ADAPTIVE_PROFILE_SCHEMA_VERSION &&
      isProfileDimension(entry.dimension) &&
      entry.subject === profileSubjectForDimension(entry.dimension) &&
      ['UNKNOWN', 'LEARNING', 'KNOWN'].includes(entry.state) &&
      [
        'OWNER_CORRECTION',
        'OWNER_EXPLICIT',
        'HOUSEHOLD_EXPLICIT',
        'HISTORICAL_QUIZ',
        'OUTCOME_INFERENCE',
      ].includes(entry.provenance) &&
      Number.isFinite(entry.confidence) &&
      entry.confidence >= 0 &&
      entry.confidence <= 1 &&
      Number.isFinite(entry.occurredAt.getTime())
  );

  const sorted = [...valid].sort((left, right) => {
    const dimension = left.dimension.localeCompare(right.dimension);
    if (dimension !== 0) return dimension;

    const leftProvenance = left.provenance as ProfileEvidenceProvenance;
    const rightProvenance = right.provenance as ProfileEvidenceProvenance;
    const provenance = PROVENANCE_RANK[rightProvenance] - PROVENANCE_RANK[leftProvenance];
    if (provenance !== 0) return provenance;

    // Records are append-only snapshots. Within the same authority class, the latest
    // explicit statement supersedes the older one, including an explicit UNKNOWN
    // correction that clears a previously known value.
    const occurredAt = right.occurredAt.getTime() - left.occurredAt.getTime();
    if (occurredAt !== 0) return occurredAt;

    const leftState = left.state as ProfileEvidence['state'];
    const rightState = right.state as ProfileEvidence['state'];
    const state = STATE_RANK[rightState] - STATE_RANK[leftState];
    if (state !== 0) return state;
    if (right.confidence !== left.confidence) return right.confidence - left.confidence;
    return left.id.localeCompare(right.id);
  });

  const byDimension = new Map<ProfileDimension, PersistedProfileEvidenceLike>();
  for (const entry of sorted) {
    const dimension = entry.dimension as ProfileDimension;
    if (!byDimension.has(dimension)) byDimension.set(dimension, entry);
  }

  return PROFILE_DIMENSIONS.flatMap((dimension) => {
    const entry = byDimension.get(dimension);
    return entry ? [entry] : [];
  });
}

export function toPolicyEvidence(entry: PersistedProfileEvidenceLike): ProfileEvidence | null {
  if (entry.schemaVersion !== ADAPTIVE_PROFILE_SCHEMA_VERSION) return null;
  if (!isProfileDimension(entry.dimension)) return null;
  if (!['UNKNOWN', 'LEARNING', 'KNOWN'].includes(entry.state)) return null;
  if (
    ![
      'OWNER_CORRECTION',
      'OWNER_EXPLICIT',
      'HOUSEHOLD_EXPLICIT',
      'HISTORICAL_QUIZ',
      'OUTCOME_INFERENCE',
    ].includes(entry.provenance)
  ) {
    return null;
  }
  return {
    dimension: entry.dimension,
    state: entry.state as ProfileEvidence['state'],
    confidence: entry.confidence,
    provenance: entry.provenance as ProfileEvidenceProvenance,
    updatedAt: entry.occurredAt.toISOString(),
  };
}

export function profileQuestionPolicyVersion(): string {
  return PROFILE_QUESTION_POLICY_V1.version;
}
