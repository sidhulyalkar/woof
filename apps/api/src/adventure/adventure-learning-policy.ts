import { WELLBEING_PATHWAYS, type WellbeingPathway } from '../care-events/care-event.types';

export const ADVENTURE_LEARNING_POLICY_VERSION = 'adventure-learning-v2';

const DAY_MS = 24 * 60 * 60 * 1000;
const DURABLE_WINDOW_DAYS = 28;
const TEMPORARY_WINDOW_DAYS = 3;

export type AdventureLearningEvent = {
  id: string;
  eventType: string;
  pathway: WellbeingPathway;
  occurredAt: string;
  context: Record<string, unknown> | null;
  outcome: Record<string, unknown> | null;
};

export type AdventureLearningSignals = {
  policyVersion: typeof ADVENTURE_LEARNING_POLICY_VERSION;
  durablePathwayPreference: Partial<Record<WellbeingPathway, number>>;
  temporaryPathwayModifier: Partial<Record<WellbeingPathway, number>>;
  temporaryPace: 'normal' | 'easy';
};

export function deriveAdventureLearningSignals(
  events: AdventureLearningEvent[],
  now = new Date()
): AdventureLearningSignals {
  const durable: Partial<Record<WellbeingPathway, number>> = {};
  const temporary: Partial<Record<WellbeingPathway, number>> = {};
  let temporaryPace: 'normal' | 'easy' = 'normal';

  const canonical = [...events]
    .filter((event) => Number.isFinite(new Date(event.occurredAt).getTime()))
    .sort(
      (left, right) =>
        new Date(right.occurredAt).getTime() - new Date(left.occurredAt).getTime() ||
        left.id.localeCompare(right.id)
    )
    .slice(0, 24);

  for (const event of canonical) {
    const ageDays = Math.max(0, (now.getTime() - new Date(event.occurredAt).getTime()) / DAY_MS);
    const outcome = event.outcome ?? {};
    const dogExperience = stringValue(outcome.dogExperience);
    const ownerExperience = stringValue(outcome.ownerExperience);
    const safeOptOut = outcome.safeOptOut === true;
    const originalPathway = learningPathway(event);

    // Durable preference is dog-level evidence only. Owner load and welfare-respecting
    // opt-outs are context, not a durable statement about what this dog likes.
    if (
      originalPathway &&
      ageDays <= DURABLE_WINDOW_DAYS &&
      ownerExperience !== 'a_lot_today' &&
      !safeOptOut
    ) {
      const recency = Math.max(0.2, 1 - ageDays / DURABLE_WINDOW_DAYS);
      const delta = durableDelta(dogExperience) * recency;
      addSignal(durable, originalPathway, delta, -0.08, 0.08);
    }

    if (ageDays > TEMPORARY_WINDOW_DAYS) continue;
    const temporaryRecency = Math.max(0, 1 - ageDays / TEMPORARY_WINDOW_DAYS);

    if (ownerExperience === 'a_lot_today') {
      temporaryPace = ageDays <= 1 ? 'easy' : temporaryPace;
      for (const pathway of ['MOVE', 'EXPLORE', 'ENRICH', 'LEARN', 'CONNECT'] as const) {
        addSignal(temporary, pathway, -0.018 * temporaryRecency, -0.06, 0.06);
      }
      addSignal(temporary, 'RECOVER', 0.03 * temporaryRecency, -0.06, 0.06);
      addSignal(temporary, 'BOND', 0.012 * temporaryRecency, -0.06, 0.06);
    }

    if (safeOptOut) {
      temporaryPace = ageDays <= 1 ? 'easy' : temporaryPace;
      if (originalPathway) {
        addSignal(temporary, originalPathway, -0.028 * temporaryRecency, -0.06, 0.06);
      }
      addSignal(temporary, 'RECOVER', 0.025 * temporaryRecency, -0.06, 0.06);
      addSignal(temporary, 'BOND', 0.01 * temporaryRecency, -0.06, 0.06);
    }
  }

  return {
    policyVersion: ADVENTURE_LEARNING_POLICY_VERSION,
    durablePathwayPreference: stableRecord(durable),
    temporaryPathwayModifier: stableRecord(temporary),
    temporaryPace,
  };
}

function learningPathway(event: AdventureLearningEvent): WellbeingPathway | null {
  const original = event.context?.originalPathway;
  if (isWellbeingPathway(original)) return original;

  // A mismatch can intentionally earn a BOND reward. Without original-pathway
  // provenance, treating that reward pathway as the disliked target would corrupt
  // long-term Bond preference. Ignore that legacy ambiguity rather than guess.
  if (event.pathway === 'BOND' && event.outcome?.dogExperience === 'not_their_thing') {
    return null;
  }

  return event.pathway;
}

function durableDelta(dogExperience: string | null) {
  if (dogExperience === 'loved_it') return 0.018;
  if (dogExperience === 'comfortable') return 0.01;
  if (dogExperience === 'not_their_thing') return -0.025;
  return 0;
}

function isWellbeingPathway(value: unknown): value is WellbeingPathway {
  return (
    typeof value === 'string' &&
    WELLBEING_PATHWAYS.includes(value as (typeof WELLBEING_PATHWAYS)[number])
  );
}

function stringValue(value: unknown) {
  return typeof value === 'string' ? value : null;
}

function addSignal(
  signals: Partial<Record<WellbeingPathway, number>>,
  pathway: WellbeingPathway,
  delta: number,
  min: number,
  max: number
) {
  if (!Number.isFinite(delta) || delta === 0) return;
  signals[pathway] = clamp((signals[pathway] ?? 0) + delta, min, max);
}

function stableRecord(signals: Partial<Record<WellbeingPathway, number>>) {
  return Object.fromEntries(
    WELLBEING_PATHWAYS.filter((pathway) => signals[pathway] !== undefined).map((pathway) => [
      pathway,
      round4(signals[pathway] ?? 0),
    ])
  ) as Partial<Record<WellbeingPathway, number>>;
}

function round4(value: number) {
  return Math.round(value * 10_000) / 10_000;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
