import {
  BEHAVIOR_DIMENSIONS,
  BEHAVIOR_PROFILE_SCHEMA_VERSION,
  type BehaviorDimension,
  type BehaviorDimensionEstimate,
  type HandlerAction,
  type IndividualBehaviorProfile,
  type InterventionEffect,
  type StoredBehaviorObservation,
} from './behavior-vision.types';

const SAFE_EXPERIMENT_ACTIONS = new Set<HandlerAction>([
  'increase-distance',
  'loosen-leash',
  'single-cue',
  'find-it',
  'parallel-walk',
  'u-turn',
  'pause-and-observe',
  'end-interaction',
]);

function clamp01(value: number) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function dimensionValue(
  observation: StoredBehaviorObservation,
  dimension: BehaviorDimension
): BehaviorDimensionEstimate | null {
  const estimate = observation.analysis.dimensions.find((entry) => entry.dimension === dimension);
  if (!estimate) return null;
  return {
    ...estimate,
    value: clamp01(estimate.value),
    confidence: clamp01(estimate.confidence),
  };
}

function observationWeight(observation: StoredBehaviorObservation) {
  if (observation.analysis.releaseQualification?.qualified !== true) return 0;
  if (!observation.analysis.mediaQuality.usable) return 0;
  if (observation.ownerFeedback?.accurate === false) return 0;
  const modelConfidence = clamp01(observation.analysis.mediaQuality.confidence);
  const feedbackMultiplier = observation.ownerFeedback?.accurate === true ? 1.15 : 1;
  return clamp01(modelConfidence * feedbackMultiplier);
}

function deriveBaselines(observations: StoredBehaviorObservation[]) {
  return BEHAVIOR_DIMENSIONS.map((dimension) => {
    const values = observations
      .map((observation) => {
        const estimate = dimensionValue(observation, dimension);
        const weight = observationWeight(observation) * (estimate?.confidence ?? 0);
        return estimate && weight > 0 ? { value: estimate.value, weight } : null;
      })
      .filter((entry): entry is { value: number; weight: number } => entry !== null);

    const totalWeight = values.reduce((sum, entry) => sum + entry.weight, 0);
    const mean = totalWeight
      ? values.reduce((sum, entry) => sum + entry.value * entry.weight, 0) / totalWeight
      : 0;
    const averageWeight = values.length ? totalWeight / values.length : 0;
    const evidenceCoverage = Math.min(1, Math.sqrt(values.length) / 3.5);

    return {
      dimension,
      mean: clamp01(mean),
      confidence: clamp01(averageWeight * evidenceCoverage),
      sampleCount: values.length,
    };
  }).filter((entry) => entry.sampleCount > 0);
}

type PairedDelta = {
  action: HandlerAction;
  arousalDelta: number | null;
  tensionDelta: number | null;
  engagementDelta: number | null;
  weight: number;
};

function deltaFor(
  baseline: StoredBehaviorObservation,
  after: StoredBehaviorObservation,
  dimension: BehaviorDimension
) {
  const before = dimensionValue(baseline, dimension);
  const next = dimensionValue(after, dimension);
  if (!before || !next) return null;
  if (before.confidence < 0.35 || next.confidence < 0.35) return null;
  return next.value - before.value;
}

function derivePairedDeltas(observations: StoredBehaviorObservation[]): PairedDelta[] {
  const sessions = new Map<string, StoredBehaviorObservation[]>();

  for (const observation of observations) {
    const key = observation.context.sessionKey;
    if (!key || observationWeight(observation) === 0) continue;
    const group = sessions.get(key) ?? [];
    group.push(observation);
    sessions.set(key, group);
  }

  const deltas: PairedDelta[] = [];
  for (const group of sessions.values()) {
    const ordered = [...group].sort(
      (left, right) => new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime()
    );
    const baseline = ordered.find((entry) => entry.context.phase === 'baseline');
    if (!baseline) continue;

    const after = ordered.find(
      (entry) =>
        entry.context.phase !== 'baseline' &&
        entry.context.handlerAction !== 'none' &&
        entry.context.handlerAction !== 'allow-greeting'
    );
    if (!after) continue;

    deltas.push({
      action: after.context.handlerAction,
      arousalDelta: deltaFor(baseline, after, 'arousal'),
      tensionDelta: deltaFor(baseline, after, 'body-tension'),
      engagementDelta: deltaFor(baseline, after, 'handler-engagement'),
      weight: Math.min(observationWeight(baseline), observationWeight(after)),
    });
  }

  return deltas;
}

function weightedNullableMean(
  entries: PairedDelta[],
  selector: (entry: PairedDelta) => number | null
) {
  const available = entries
    .map((entry) => ({ value: selector(entry), weight: entry.weight }))
    .filter((entry): entry is { value: number; weight: number } => entry.value !== null);
  const totalWeight = available.reduce((sum, entry) => sum + entry.weight, 0);
  if (!totalWeight) return null;
  return available.reduce((sum, entry) => sum + entry.value * entry.weight, 0) / totalWeight;
}

function deriveInterventionEffects(
  observations: StoredBehaviorObservation[]
): InterventionEffect[] {
  const deltas = derivePairedDeltas(observations);
  const byAction = new Map<HandlerAction, PairedDelta[]>();
  for (const delta of deltas) {
    const current = byAction.get(delta.action) ?? [];
    current.push(delta);
    byAction.set(delta.action, current);
  }

  return [...byAction.entries()]
    .map(([action, entries]) => {
      const avgWeight = entries.reduce((sum, entry) => sum + entry.weight, 0) / entries.length;
      return {
        action,
        pairedSessions: entries.length,
        arousalDelta: weightedNullableMean(entries, (entry) => entry.arousalDelta),
        tensionDelta: weightedNullableMean(entries, (entry) => entry.tensionDelta),
        engagementDelta: weightedNullableMean(entries, (entry) => entry.engagementDelta),
        confidence: clamp01(avgWeight * Math.min(1, Math.sqrt(entries.length) / 2.5)),
      };
    })
    .sort((left, right) => right.confidence - left.confidence);
}

function effectScore(effect: InterventionEffect) {
  const arousalBenefit = effect.arousalDelta === null ? 0 : -effect.arousalDelta;
  const tensionBenefit = effect.tensionDelta === null ? 0 : -effect.tensionDelta;
  const engagementBenefit = effect.engagementDelta ?? 0;
  return (arousalBenefit + tensionBenefit + 0.5 * engagementBenefit) * effect.confidence;
}

function getMean(
  profile: ReturnType<typeof deriveBaselines>,
  dimension: BehaviorDimension
): number | null {
  return profile.find((entry) => entry.dimension === dimension)?.mean ?? null;
}

function buildRecommendation(input: {
  baselines: ReturnType<typeof deriveBaselines>;
  effects: InterventionEffect[];
  sampleCount: number;
  contextCount: number;
}) {
  const { baselines, effects, sampleCount, contextCount } = input;
  const arousal = getMean(baselines, 'arousal');
  const socialOrientation = getMean(baselines, 'social-orientation');
  const approach = getMean(baselines, 'approach-tendency');
  const avoidance = getMean(baselines, 'avoidance-tendency');

  const bestSafeEffect = effects
    .filter(
      (effect) =>
        SAFE_EXPERIMENT_ACTIONS.has(effect.action) &&
        effect.pairedSessions >= 2 &&
        effect.confidence >= 0.3 &&
        effectScore(effect) > 0.03
    )
    .sort((left, right) => effectScore(right) - effectScore(left))[0];

  if (bestSafeEffect) {
    return {
      headline: 'Repeat the handler strategy that has helped this dog recover',
      explanation: `Across ${bestSafeEffect.pairedSessions} paired observations, “${bestSafeEffect.action}” is associated with a calmer or more engaged response for this individual dog. This is an N-of-1 association, not proof of cause, so keep testing it in easy contexts.`,
      nextSafeExperiment: [
        'Start far enough from the trigger that the dog can still observe and disengage.',
        `Repeat “${bestSafeEffect.action}” while keeping every other part of the setup as similar as practical.`,
        'Record a short baseline clip and a recovery clip so Woof can compare the response within the same session.',
      ],
      neverAutoRecommendGreeting: true as const,
    };
  }

  if (
    arousal !== null &&
    socialOrientation !== null &&
    approach !== null &&
    arousal >= 0.6 &&
    socialOrientation >= 0.6 &&
    approach >= 0.55 &&
    (avoidance ?? 0) < 0.65
  ) {
    return {
      headline: 'High social orientation does not automatically mean “let them greet”',
      explanation:
        'This dog often orients or moves toward other dogs while arousal is elevated. That pattern can occur with excitement, barrier frustration, uncertainty, learned leash behavior, or mixed motivation. Woof should test recovery and body looseness before making any social-access interpretation.',
      nextSafeExperiment: [
        'Observe another dog from a distance where this dog can still take food, sniff, or disengage.',
        'Try a parallel path or increase distance instead of moving directly toward the other dog.',
        'Compare recovery after one calm handler action rather than repeating cues or holding the leash tight.',
      ],
      neverAutoRecommendGreeting: true as const,
    };
  }

  if (sampleCount < 5 || contextCount < 2) {
    return {
      headline: 'Woof is still learning this dog',
      explanation:
        'A few clips can describe visible behavior, but they are not enough to decide what reliably helps this individual. Collect short observations in more than one context before treating any pattern as personal baseline.',
      nextSafeExperiment: [
        'Capture 10–20 seconds before changing anything.',
        'Make one low-risk change such as adding distance, loosening leash tension, or pausing.',
        'Capture a second short clip during recovery and tell Woof whether its observations were accurate.',
      ],
      neverAutoRecommendGreeting: true as const,
    };
  }

  return {
    headline: 'Keep learning from repeated, comparable situations',
    explanation:
      'The current evidence does not support one clearly better handler strategy yet. Continue changing one variable at a time and let the individual dog’s observed response outrank generic breed or behavior assumptions.',
    nextSafeExperiment: [
      'Use a familiar, low-intensity setup.',
      'Change only one variable such as distance, movement, leash slack, or cue frequency.',
      'Record whether recovery becomes faster, body tension changes, and voluntary engagement returns.',
    ],
    neverAutoRecommendGreeting: true as const,
  };
}

export function deriveIndividualBehaviorProfile(
  petId: string,
  observations: StoredBehaviorObservation[]
): IndividualBehaviorProfile {
  const usable = observations.filter((observation) => observationWeight(observation) > 0);
  const contextsSeen = [
    ...new Set(usable.map((observation) => observation.context.context)),
  ].sort();
  const baselines = deriveBaselines(usable);
  const interventionEffects = deriveInterventionEffects(usable);
  const averageBaselineConfidence = baselines.length
    ? baselines.reduce((sum, baseline) => sum + baseline.confidence, 0) / baselines.length
    : 0;
  const breadth = Math.min(1, contextsSeen.length / 4);
  const depth = Math.min(1, Math.sqrt(usable.length) / 4);
  const personalizationConfidence = clamp01(
    averageBaselineConfidence * 0.5 + breadth * 0.2 + depth * 0.3
  );

  return {
    schemaVersion: BEHAVIOR_PROFILE_SCHEMA_VERSION,
    petId,
    sampleCount: usable.length,
    contextsSeen,
    baselines,
    interventionEffects,
    personalizationConfidence,
    recommendation: buildRecommendation({
      baselines,
      effects: interventionEffects,
      sampleCount: usable.length,
      contextCount: contextsSeen.length,
    }),
  };
}
