export const PROFILE_DIMENSIONS = [
  'OWNER_GOALS',
  'OWNER_TIME_BUDGET',
  'OWNER_EFFORT_PREFERENCE',
  'AVAILABLE_ENVIRONMENTS',
  'DOG_ENERGY_PATTERN',
  'DOG_SOCIAL_COMFORT',
  'DOG_NOVELTY_COMFORT',
  'DOG_REINFORCERS',
  'DOG_OBVIOUS_DISLIKES',
  'TRAINING_EXPERIENCE',
] as const;

export type ProfileDimension = (typeof PROFILE_DIMENSIONS)[number];
export type ProfileEvidenceState = 'UNKNOWN' | 'LEARNING' | 'KNOWN';
export type ProfileEvidenceProvenance =
  | 'OWNER_CORRECTION'
  | 'OWNER_EXPLICIT'
  | 'HOUSEHOLD_EXPLICIT'
  | 'HISTORICAL_QUIZ'
  | 'OUTCOME_INFERENCE';

export type ProfileEvidence = {
  dimension: ProfileDimension;
  state: ProfileEvidenceState;
  confidence: number;
  provenance: ProfileEvidenceProvenance;
  updatedAt: string;
};

export type QuestionHistoryOutcome = 'ANSWERED' | 'NOT_SURE' | 'SKIPPED';

export type QuestionHistoryEntry = {
  questionId: string;
  askedAt: string;
  outcome: QuestionHistoryOutcome;
};

export type QuestFamily =
  'ACTIVITY' | 'SCENT' | 'TRAINING' | 'SOCIAL' | 'RECOVERY' | 'BOND' | 'OTHER';

export type ProfileInteraction = {
  id: string;
  occurredAt: string;
  kind: 'DISMISSED' | 'COMPLETED' | 'SAFE_OPT_OUT';
  questFamily: QuestFamily;
  dogExperience?: 'loved_it' | 'comfortable' | 'not_their_thing';
};

export type ProfileQuestionTarget =
  | { kind: 'PROFILE'; dimension: ProfileDimension }
  | { kind: 'QUEST_FAMILY_PREFERENCE'; questFamily: QuestFamily }
  | { kind: 'SESSION_DIFFICULTY'; questFamily: 'TRAINING' };

export type ProfileQuestion = {
  policyVersion: string;
  id: string;
  target: ProfileQuestionTarget;
  prompt: string;
  whyAsk: string;
  answers: readonly string[];
  score: number;
  reasonCodes: readonly string[];
};

export type ProfileQuestionPolicyInput = {
  profileEvidence: readonly ProfileEvidence[];
  questionHistory: readonly QuestionHistoryEntry[];
  interactions: readonly ProfileInteraction[];
  now: string;
};

export type ProfileQuestionPolicyReceipt = {
  version: string;
  knownConfidenceThreshold: number;
  interactionWindowDays: number;
  sessionFollowupWindowHours: number;
  answeredCooldownDays: number;
  notSureCooldownDays: number;
  skippedCooldownDays: number;
  dismissalsForClarification: number;
  scoreWeights: {
    decisionValue: number;
    uncertainty: number;
    contextTrigger: number;
    safetyRelevance: number;
    burden: number;
  };
};
