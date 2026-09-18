import type { WellbeingPathway } from '../care-events/care-event.types';
import { ADVENTURE_LEARNING_POLICY_VERSION } from './adventure-learning-policy';
import type { CompleteQuestDto } from './dto/adventure.dto';

export type AdventureCompletionOutcome = {
  dogExperience: CompleteQuestDto['dogExperience'];
  ownerExperience: CompleteQuestDto['ownerExperience'];
  safeOptOut: boolean;
};

export type AdventureLearningReceipt = {
  policyVersion: typeof ADVENTURE_LEARNING_POLICY_VERSION;
  headline: string;
  dogSignal: string;
  humanSignal: string | null;
  nextRecommendationEffect: string;
  qualifier: string;
};

type AdventureLearningReceiptInput = AdventureCompletionOutcome & {
  pathway: WellbeingPathway;
};

const DOG_EXPERIENCES = ['loved_it', 'comfortable', 'not_their_thing'] as const;
const OWNER_EXPERIENCES = ['great', 'fine', 'a_lot_today'] as const;

const PATHWAY_COPY: Record<WellbeingPathway, string> = {
  MOVE: 'movement',
  EXPLORE: 'exploration',
  ENRICH: 'enrichment',
  LEARN: 'learning',
  CONNECT: 'social',
  CARE: 'care',
  RECOVER: 'recovery',
  BOND: 'bonding',
};

export function parseCanonicalAdventureOutcome(
  value: Record<string, unknown> | null | undefined
): AdventureCompletionOutcome | null {
  if (!value) return null;

  const dogExperience = value.dogExperience;
  const ownerExperience = value.ownerExperience;
  const safeOptOut = value.safeOptOut;

  if (
    !DOG_EXPERIENCES.includes(dogExperience as AdventureCompletionOutcome['dogExperience']) ||
    !OWNER_EXPERIENCES.includes(ownerExperience as AdventureCompletionOutcome['ownerExperience']) ||
    (safeOptOut !== undefined && typeof safeOptOut !== 'boolean')
  ) {
    return null;
  }

  return {
    dogExperience: dogExperience as AdventureCompletionOutcome['dogExperience'],
    ownerExperience: ownerExperience as AdventureCompletionOutcome['ownerExperience'],
    safeOptOut: safeOptOut === true,
  };
}

/**
 * Present the exact bounded semantics already owned by adventure-learning-policy.
 *
 * This function intentionally does not reproduce numeric score deltas. It translates
 * canonical outcome semantics into user-safe language while preserving the policy's
 * important separations:
 * - dog fit can become bounded durable pathway evidence;
 * - owner load is temporary context, never dog preference;
 * - a safe opt-out is temporary pacing evidence and never durable dislike.
 */
export function buildAdventureLearningReceipt(
  input: AdventureLearningReceiptInput
): AdventureLearningReceipt {
  const pathway = PATHWAY_COPY[input.pathway];
  const ownerHadHighLoad = input.ownerExperience === 'a_lot_today';

  if (input.safeOptOut) {
    return {
      policyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
      headline: 'Good read.',
      dogSignal: `Stopping is not counted as lasting evidence that ${pathway} is a poor fit.`,
      humanSignal: ownerHadHighLoad
        ? "You also said this was a lot for you. Woof keeps your load separate from your dog's preference."
        : null,
      nextRecommendationEffect:
        'For the next few days, Woof will lean gentler and give recovery more room.',
      qualifier:
        'Respecting an exit is a successful outcome. This stop is not treated as a permanent preference judgment.',
    };
  }

  const negativeFit = input.dogExperience === 'not_their_thing';
  const dogSignal = negativeFit
    ? `This adds a bounded negative fit signal for ${pathway}.`
    : `This adds a small positive fit signal for ${pathway}.`;

  let nextRecommendationEffect: string;
  if (ownerHadHighLoad && negativeFit) {
    nextRecommendationEffect = `For the next few days, Woof will lean easier and give similar ${pathway} ideas a little less weight while it watches for a pattern.`;
  } else if (ownerHadHighLoad) {
    nextRecommendationEffect = `This can remain a ${pathway} fit clue while Woof temporarily leans toward an easier pace.`;
  } else if (negativeFit) {
    nextRecommendationEffect = `Similar ${pathway} ideas may be gently deprioritized while Woof watches for a pattern.`;
  } else {
    nextRecommendationEffect = `Similar ${pathway} ideas may be gently favored while Woof watches for a pattern.`;
  }

  return {
    policyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
    headline:
      input.dogExperience === 'loved_it'
        ? 'Worth remembering.'
        : input.dogExperience === 'comfortable'
          ? 'A useful comfortable data point.'
          : 'Useful discovery, not a failure.',
    dogSignal,
    humanSignal: ownerHadHighLoad
      ? "You said this was a lot for you. Woof keeps your load separate from your dog's preference."
      : null,
    nextRecommendationEffect,
    qualifier: 'One session is a clue, not a permanent preference.',
  };
}
