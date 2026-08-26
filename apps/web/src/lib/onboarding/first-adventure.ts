import type { AdaptiveProfileQuestionResponseInput } from '../api/adaptive-profile';

export const FIRST_ADVENTURE_GOALS = [
  'MORE_ADVENTURES',
  'TRAINING',
  'CALMER_ROUTINES',
  'SOCIAL_CONFIDENCE',
  'CARE_ROUTINES',
  'JUST_HAVE_FUN',
] as const;

export const FIRST_ADVENTURE_TIME_BUDGETS = [
  'FIVE_MIN',
  'TEN_TO_FIFTEEN',
  'TWENTY_TO_THIRTY',
  'FORTY_PLUS',
  'VARIES',
] as const;

export const FIRST_ADVENTURE_EFFORT_LEVELS = [
  'KEEP_IT_EASY',
  'MODERATE',
  'UP_FOR_A_CHALLENGE',
  'VARIES',
] as const;

export const FIRST_ADVENTURE_SOCIAL_COMFORT = [
  'PREFERS_SPACE',
  'CALM_AT_DISTANCE',
  'SELECTIVELY_SOCIAL',
  'OFTEN_SOCIAL',
  'NOT_SURE',
] as const;

export type FirstAdventureGoal = (typeof FIRST_ADVENTURE_GOALS)[number];
export type FirstAdventureTimeBudget = (typeof FIRST_ADVENTURE_TIME_BUDGETS)[number];
export type FirstAdventureEffort = (typeof FIRST_ADVENTURE_EFFORT_LEVELS)[number];
export type FirstAdventureSocialComfort = (typeof FIRST_ADVENTURE_SOCIAL_COMFORT)[number];

export type FirstAdventureSelections = {
  goals: FirstAdventureGoal[];
  timeBudget: FirstAdventureTimeBudget | null;
  effort: FirstAdventureEffort | null;
  socialComfort: FirstAdventureSocialComfort | null;
};

const QUESTION_IDS = {
  goals: 'profile-owner-goals-v1',
  timeBudget: 'profile-owner-time-budget-v1',
  effort: 'profile-owner-effort-v1',
  socialComfort: 'profile-dog-social-comfort-v1',
} as const;

function responseId(petId: string, questionId: string) {
  return `first-adventure-v1:${petId}:${questionId}`;
}

function skipped(petId: string, questionId: string): AdaptiveProfileQuestionResponseInput {
  return {
    responseId: responseId(petId, questionId),
    questionId,
    outcome: 'SKIPPED',
  };
}

export function buildFirstAdventureResponses(
  petId: string,
  selections: FirstAdventureSelections,
  skipAll = false
): AdaptiveProfileQuestionResponseInput[] {
  if (selections.goals.length > 3) {
    throw new Error('First Adventure accepts at most three goals');
  }

  const goals = [...new Set(selections.goals)].sort();

  if (skipAll) {
    return Object.values(QUESTION_IDS).map((questionId) => skipped(petId, questionId));
  }

  const responses: AdaptiveProfileQuestionResponseInput[] = [];

  responses.push(
    goals.length
      ? {
          responseId: responseId(petId, QUESTION_IDS.goals),
          questionId: QUESTION_IDS.goals,
          outcome: 'ANSWERED',
          answers: goals,
        }
      : skipped(petId, QUESTION_IDS.goals)
  );

  responses.push(
    selections.timeBudget
      ? {
          responseId: responseId(petId, QUESTION_IDS.timeBudget),
          questionId: QUESTION_IDS.timeBudget,
          outcome: 'ANSWERED',
          answers: [selections.timeBudget],
        }
      : skipped(petId, QUESTION_IDS.timeBudget)
  );

  responses.push(
    selections.effort
      ? {
          responseId: responseId(petId, QUESTION_IDS.effort),
          questionId: QUESTION_IDS.effort,
          outcome: 'ANSWERED',
          answers: [selections.effort],
        }
      : skipped(petId, QUESTION_IDS.effort)
  );

  if (selections.socialComfort === 'NOT_SURE') {
    responses.push({
      responseId: responseId(petId, QUESTION_IDS.socialComfort),
      questionId: QUESTION_IDS.socialComfort,
      outcome: 'NOT_SURE',
    });
  } else if (selections.socialComfort) {
    responses.push({
      responseId: responseId(petId, QUESTION_IDS.socialComfort),
      questionId: QUESTION_IDS.socialComfort,
      outcome: 'ANSWERED',
      answers: [selections.socialComfort],
    });
  } else {
    responses.push(skipped(petId, QUESTION_IDS.socialComfort));
  }

  return responses;
}

export function emptyFirstAdventureSelections(): FirstAdventureSelections {
  return {
    goals: [],
    timeBudget: null,
    effort: null,
    socialComfort: null,
  };
}
