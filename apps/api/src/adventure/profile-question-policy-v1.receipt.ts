import type { ProfileQuestionPolicyReceipt } from './profile-question-policy-v1.types';

export const PROFILE_QUESTION_POLICY_V1: ProfileQuestionPolicyReceipt = Object.freeze({
  version: 'profile-question-policy-v1',
  knownConfidenceThreshold: 0.8,
  interactionWindowDays: 14,
  sessionFollowupWindowHours: 24,
  answeredCooldownDays: 7,
  notSureCooldownDays: 7,
  skippedCooldownDays: 14,
  dismissalsForClarification: 2,
  scoreWeights: Object.freeze({
    decisionValue: 3,
    uncertainty: 2,
    contextTrigger: 2,
    safetyRelevance: 1,
    burden: -1,
  }),
});

export const PROFILE_QUESTION_POLICY_V1_CANONICAL_RECEIPT =
  '{"answeredCooldownDays":7,"dismissalsForClarification":2,"interactionWindowDays":14,"knownConfidenceThreshold":0.8,"notSureCooldownDays":7,"scoreWeights":{"burden":-1,"contextTrigger":2,"decisionValue":3,"safetyRelevance":1,"uncertainty":2},"sessionFollowupWindowHours":24,"skippedCooldownDays":14,"version":"profile-question-policy-v1"}';

export const PROFILE_QUESTION_POLICY_V1_SHA256 =
  'dca2936d547338e58d5911006e575d8394d685e7679c2d52ad5cb24e07cd0483';
