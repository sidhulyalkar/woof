import type { CareEventInput, WellbeingPathway } from './care-event.types';

export const REWARD_POLICY_VERSION = 'bond-xp-v1';
export const DAILY_BOND_XP_CAP = 120;
export const DAILY_PATHWAY_XP_CAP = 60;

const BASE_XP: Record<string, number> = {
  QUEST_MOVE: 15,
  QUEST_EXPLORE: 20,
  QUEST_ENRICH: 15,
  QUEST_LEARN: 15,
  QUEST_CONNECT: 20,
  QUEST_CARE: 18,
  QUEST_RECOVER: 12,
  QUEST_BOND: 12,
  ACTIVITY_WALK: 15,
  ACTIVITY_RUN: 22,
  ACTIVITY_HIKE: 30,
  ACTIVITY_PLAY: 15,
  ACTIVITY_SNIFF: 20,
  TRAINING_SESSION: 15,
  ENRICHMENT_SESSION: 15,
  SOCIAL_OUTING: 20,
  PARALLEL_WALK: 20,
  WELLNESS_VISIT: 40,
  DENTAL_CARE: 8,
  COOPERATIVE_CARE: 15,
  RECOVERY_SESSION: 12,
  QUEST_REFLECTION: 5,
  SAFE_OPT_OUT: 18,
  MEMORY_ADDED: 2,
};

export type RewardPolicyContext = {
  totalXpToday: number;
  pathwayXpToday: number;
  samePathwayEventsToday: number;
  repeatedEventCount7d: number;
};

export type RewardDecision = {
  bondXp: number;
  pathwayXp: Partial<Record<WellbeingPathway, number>>;
  policyVersion: string;
  explanation: string;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function booleanContext(input: CareEventInput, key: string) {
  return input.context?.[key] === true || input.outcome?.[key] === true;
}

export function rewardCareEvent(
  input: CareEventInput,
  context: RewardPolicyContext,
): RewardDecision {
  if (input.safetyEligible === false) {
    return {
      bondXp: 0,
      pathwayXp: {},
      policyVersion: REWARD_POLICY_VERSION,
      explanation: 'No reward was issued because the event was not safety-eligible.',
    };
  }

  const base = BASE_XP[input.eventType] ?? 0;
  if (base <= 0) {
    return {
      bondXp: 0,
      pathwayXp: {},
      policyVersion: REWARD_POLICY_VERSION,
      explanation: 'This event is useful context but is not eligible for Bond XP.',
    };
  }

  // Evidence changes confidence, not the moral value of the action. The multiplier
  // deliberately stays in a narrow range so photos, wearables, or location never
  // dominate self-report.
  const evidenceConfidence = clamp(input.evidenceConfidence ?? 0.65, 0, 1);
  const evidenceMultiplier = 0.96 + evidenceConfidence * 0.04;

  // Repeating the same pathway remains useful, but the game does not reward farming.
  const diversityMultiplier =
    context.samePathwayEventsToday <= 1
      ? 1
      : context.samePathwayEventsToday === 2
        ? 0.82
        : 0.58;
  const repetitionMultiplier =
    context.repeatedEventCount7d <= 2
      ? 1
      : context.repeatedEventCount7d <= 5
        ? 0.88
        : 0.72;

  // Quest ranking may contribute a server-generated relevance value. It is tightly
  // capped and never accepted as an arbitrary XP amount.
  const relevance = Number(input.context?.personalRelevance ?? 1);
  const relevanceMultiplier = clamp(Number.isFinite(relevance) ? relevance : 1, 0.9, 1.08);

  let raw = base * evidenceMultiplier * diversityMultiplier * repetitionMultiplier * relevanceMultiplier;

  const reflected =
    typeof input.outcome?.dogExperience === 'string' &&
    typeof input.outcome?.ownerExperience === 'string';
  if (reflected && input.eventType !== 'QUEST_REFLECTION') raw += 5;

  if (booleanContext(input, 'newPlace')) raw += 5;
  if (booleanContext(input, 'memoryAdded')) raw += 2;

  // Respecting a stress/opt-out signal is a successful welfare decision. SAFE_OPT_OUT
  // has its own full-value base reward rather than requiring task completion.
  if (input.eventType === 'SAFE_OPT_OUT') raw = Math.max(raw, BASE_XP.SAFE_OPT_OUT);

  const totalRemaining = Math.max(0, DAILY_BOND_XP_CAP - context.totalXpToday);
  const pathwayRemaining = Math.max(0, DAILY_PATHWAY_XP_CAP - context.pathwayXpToday);
  const bondXp = Math.max(0, Math.floor(Math.min(raw, totalRemaining, pathwayRemaining)));

  const capReason =
    bondXp === 0 && (totalRemaining === 0 || pathwayRemaining === 0)
      ? ' Daily reward coverage is already complete for this window.'
      : bondXp < Math.floor(raw)
        ? ' Reward was capped to keep volume from overpowering balance.'
        : '';

  return {
    bondXp,
    pathwayXp: bondXp > 0 ? { [input.pathway]: bondXp } : {},
    policyVersion: REWARD_POLICY_VERSION,
    explanation: `Bond XP reflects a trusted ${input.pathway.toLowerCase()} experience, with diminishing returns for repetition.${capReason}`,
  };
}

export function baseXpForEvent(eventType: string) {
  return BASE_XP[eventType] ?? 0;
}
