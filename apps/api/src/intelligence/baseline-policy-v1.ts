import { BASELINE_POLICY_V1 } from './baseline-policy-v1.receipt';
import {
  SIGNAL_DIMENSIONS,
  type BaselineDimension,
  type BaselineSummary,
  type Confidence,
  type Direction,
  type EvidenceSourceSummary,
  type Magnitude,
  type NormalizedObservation,
  type SignalDimension,
} from './baseline-policy-v1.types';

const DAY_MS = 86_400_000;
const LOCAL_DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

const DIMENSION_LABELS: Record<SignalDimension, string> = {
  APPETITE: 'appetite',
  ENERGY: 'energy',
  BATHROOM_ROUTINE: 'bathroom/routine',
  MOBILITY_COMFORT: 'mobility/comfort',
  ENGAGEMENT_SOCIAL_COMFORT: 'engagement/social comfort',
  SLEEP_REST: 'sleep/rest',
};

type EvaluationInput = {
  dimension: SignalDimension;
  observations: readonly NormalizedObservation[];
  now: string;
};

type WeightedObservation = {
  observation: NormalizedObservation;
  observedAtMs: number;
  weight: number;
};

type Support = {
  positiveWeight: number;
  negativeWeight: number;
  nearWeight: number;
  positiveSamples: number;
  negativeSamples: number;
  nearSamples: number;
};

function parseTimestamp(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function evidenceWeight(observation: NormalizedObservation): number {
  return BASELINE_POLICY_V1.reliabilityWeights[observation.reliability] * observation.confidence;
}

function canonicalize(
  observations: readonly NormalizedObservation[],
  dimension: SignalDimension,
  nowMs: number
): WeightedObservation[] {
  const windowStartMs = nowMs - BASELINE_POLICY_V1.retentionWindowDays * DAY_MS;

  const valid = observations
    .filter((observation) => observation.dimension === dimension)
    .map((observation) => ({ observation, observedAtMs: parseTimestamp(observation.observedAt) }))
    .filter(
      (entry): entry is { observation: NormalizedObservation; observedAtMs: number } =>
        entry.observedAtMs !== null &&
        entry.observedAtMs <= nowMs &&
        entry.observedAtMs >= windowStartMs &&
        LOCAL_DATE_PATTERN.test(entry.observation.localDate) &&
        Number.isFinite(entry.observation.confidence) &&
        entry.observation.confidence > 0 &&
        entry.observation.confidence <= 1
    )
    .sort((left, right) => {
      const dedupe = left.observation.dedupeKey.localeCompare(right.observation.dedupeKey);
      if (dedupe !== 0) return dedupe;
      if (left.observedAtMs !== right.observedAtMs) return left.observedAtMs - right.observedAtMs;
      return left.observation.id.localeCompare(right.observation.id);
    });

  const byDedupeKey = new Map<string, (typeof valid)[number]>();
  for (const entry of valid) {
    if (!byDedupeKey.has(entry.observation.dedupeKey)) {
      byDedupeKey.set(entry.observation.dedupeKey, entry);
    }
  }

  const deduped = [...byDedupeKey.values()];
  const supersededIds = new Set(
    deduped.flatMap((entry) =>
      entry.observation.supersedesObservationId ? [entry.observation.supersedesObservationId] : []
    )
  );

  return deduped
    .filter((entry) => !supersededIds.has(entry.observation.id))
    .map((entry) => ({
      ...entry,
      weight: evidenceWeight(entry.observation),
    }))
    .filter((entry) => Number.isFinite(entry.weight) && entry.weight > 0)
    .sort((left, right) => {
      if (left.observedAtMs !== right.observedAtMs) return left.observedAtMs - right.observedAtMs;
      return left.observation.id.localeCompare(right.observation.id);
    });
}

function weightedMean(observations: readonly WeightedObservation[]): number | null {
  let totalWeight = 0;
  let weightedValue = 0;
  for (const entry of observations) {
    totalWeight += entry.weight;
    weightedValue += entry.weight * entry.observation.deltaBucket;
  }
  if (totalWeight <= 0) return null;
  const result = weightedValue / totalWeight;
  return Number.isFinite(result) ? result : null;
}

function totalWeight(observations: readonly WeightedObservation[]): number {
  return observations.reduce((sum, entry) => sum + entry.weight, 0);
}

function supportAgainstBaseline(
  recent: readonly WeightedObservation[],
  baselineMean: number
): Support {
  const support: Support = {
    positiveWeight: 0,
    negativeWeight: 0,
    nearWeight: 0,
    positiveSamples: 0,
    negativeSamples: 0,
    nearSamples: 0,
  };

  for (const entry of recent) {
    const deviation = entry.observation.deltaBucket - baselineMean;
    if (deviation >= BASELINE_POLICY_V1.directionThreshold) {
      support.positiveWeight += entry.weight;
      support.positiveSamples += 1;
    } else if (deviation <= -BASELINE_POLICY_V1.directionThreshold) {
      support.negativeWeight += entry.weight;
      support.negativeSamples += 1;
    } else {
      support.nearWeight += entry.weight;
      support.nearSamples += 1;
    }
  }

  return support;
}

function directionFromSupport(
  baselineMean: number,
  recentMean: number,
  support: Support
): Direction {
  const directionalMax = Math.max(support.positiveWeight, support.negativeWeight);
  const directionalMin = Math.min(support.positiveWeight, support.negativeWeight);
  const hasDirectionalConflict =
    directionalMax > 0 && directionalMin / directionalMax >= BASELINE_POLICY_V1.conflictRatio;

  if (hasDirectionalConflict) return 'MIXED';

  const shift = recentMean - baselineMean;
  if (Math.abs(shift) < BASELINE_POLICY_V1.directionThreshold) return 'NEAR_BASELINE';

  if (shift > 0) {
    return support.positiveSamples >= BASELINE_POLICY_V1.minimumDirectionalSamples
      ? 'HIGHER'
      : 'UNAVAILABLE';
  }

  return support.negativeSamples >= BASELINE_POLICY_V1.minimumDirectionalSamples
    ? 'LOWER'
    : 'UNAVAILABLE';
}

function magnitudeFor(direction: Direction, shift: number): Magnitude {
  if (direction === 'UNAVAILABLE' || direction === 'MIXED') return 'UNAVAILABLE';
  if (direction === 'NEAR_BASELINE') return 'SMALL';

  const absoluteShift = Math.abs(shift);
  if (absoluteShift >= BASELINE_POLICY_V1.magnitudeThresholds.large) return 'LARGE';
  if (absoluteShift >= BASELINE_POLICY_V1.magnitudeThresholds.moderate) return 'MODERATE';
  return 'SMALL';
}

function recentAgreementSampleCount(recent: readonly WeightedObservation[]): number {
  let negative = 0;
  let neutral = 0;
  let positive = 0;

  for (const entry of recent) {
    if (entry.observation.deltaBucket < 0) negative += 1;
    else if (entry.observation.deltaBucket > 0) positive += 1;
    else neutral += 1;
  }

  return Math.max(negative, neutral, positive);
}

function confidenceFor(
  dataState: BaselineDimension['dataState'],
  baseline: readonly WeightedObservation[],
  recent: readonly WeightedObservation[]
): Confidence {
  if (dataState !== 'ESTABLISHED') return 'LOW';

  const baselineWeight = totalWeight(baseline);
  const recentEvidenceScore =
    totalWeight(recent) +
    BASELINE_POLICY_V1.confidenceAgreementBonusPerSample * recentAgreementSampleCount(recent);
  const thresholds = BASELINE_POLICY_V1.confidenceThresholds;

  if (
    baselineWeight >= thresholds.highBaselineWeight &&
    recentEvidenceScore >= thresholds.highRecentEvidenceScore
  ) {
    return 'HIGH';
  }

  if (
    baselineWeight >= thresholds.mediumBaselineWeight &&
    recentEvidenceScore >= thresholds.mediumRecentEvidenceScore
  ) {
    return 'MEDIUM';
  }

  return 'LOW';
}

function sourceSummary(
  observations: readonly WeightedObservation[],
  recentStartMs: number
): EvidenceSourceSummary[] {
  const summaries = new Map<string, EvidenceSourceSummary>();
  for (const entry of observations) {
    const sourceType = entry.observation.sourceType;
    const current = summaries.get(sourceType) ?? {
      sourceType,
      samples: 0,
      recentSamples: 0,
    };
    current.samples += 1;
    if (entry.observedAtMs >= recentStartMs) current.recentSamples += 1;
    summaries.set(sourceType, current);
  }

  return [...summaries.values()].sort((left, right) =>
    left.sourceType.localeCompare(right.sourceType)
  );
}

function explanationFor(
  dimension: SignalDimension,
  dataState: BaselineDimension['dataState'],
  direction: Direction,
  magnitude: Magnitude,
  confidence: Confidence,
  baselineSamples: number,
  recentSamples: number
): string {
  const label = DIMENSION_LABELS[dimension];

  if (dataState === 'INSUFFICIENT') {
    return `Not enough valid ${label} evidence is available to estimate this dog's baseline yet.`;
  }
  if (dataState === 'LEARNING') {
    return `Woof is still learning this dog's ${label} pattern; ${baselineSamples} baseline samples are currently eligible.`;
  }
  if (dataState === 'STALE') {
    return `The established ${label} baseline is stale because no valid evidence has arrived within ${BASELINE_POLICY_V1.staleAfterDays} days.`;
  }
  if (direction === 'UNAVAILABLE') {
    return `The ${label} baseline is established, but v1 does not have enough repeated recent directional evidence to claim a change.`;
  }
  if (direction === 'MIXED') {
    return `Recent ${label} evidence points in different directions; Woof is keeping the result mixed with ${confidence.toLowerCase()} confidence.`;
  }
  if (direction === 'NEAR_BASELINE') {
    return `Recent ${label} evidence is near the established baseline with ${confidence.toLowerCase()} confidence from ${recentSamples} recent samples.`;
  }

  const magnitudeWord =
    magnitude === 'LARGE' ? 'substantially' : magnitude === 'MODERATE' ? 'moderately' : 'slightly';
  const directionWord = direction === 'HIGHER' ? 'higher' : 'lower';
  return `Recent ${label} evidence is ${magnitudeWord} ${directionWord} than the established baseline with ${confidence.toLowerCase()} confidence.`;
}

export function evaluateBaselineDimension(input: EvaluationInput): BaselineDimension {
  const nowMs = parseTimestamp(input.now);
  if (nowMs === null) throw new Error('baseline-policy-v1 requires a valid explicit now timestamp');

  const recentStartMs = nowMs - BASELINE_POLICY_V1.recentWindowDays * DAY_MS;
  const baselineStartMs = recentStartMs - BASELINE_POLICY_V1.baselineWindowDays * DAY_MS;
  const observations = canonicalize(input.observations, input.dimension, nowMs);
  const baseline = observations.filter(
    (entry) => entry.observedAtMs >= baselineStartMs && entry.observedAtMs < recentStartMs
  );
  const recent = observations.filter(
    (entry) => entry.observedAtMs >= recentStartMs && entry.observedAtMs <= nowMs
  );

  const baselineDays = new Set(baseline.map((entry) => entry.observation.localDate)).size;
  const allDays = new Set(observations.map((entry) => entry.observation.localDate)).size;
  const latest = observations.at(-1);
  const isEstablished = baselineDays >= BASELINE_POLICY_V1.establishedMinDistinctDays;
  const isStale =
    isEstablished &&
    latest !== undefined &&
    nowMs - latest.observedAtMs > BASELINE_POLICY_V1.staleAfterDays * DAY_MS;

  const dataState: BaselineDimension['dataState'] = isStale
    ? 'STALE'
    : isEstablished
      ? 'ESTABLISHED'
      : allDays >= BASELINE_POLICY_V1.learningMinDistinctDays
        ? 'LEARNING'
        : 'INSUFFICIENT';

  const baselineMean = weightedMean(baseline);
  const recentMean = weightedMean(recent);
  let direction: Direction = 'UNAVAILABLE';
  let magnitude: Magnitude = 'UNAVAILABLE';
  const confidence = confidenceFor(dataState, baseline, recent);

  if (dataState === 'ESTABLISHED' && baselineMean !== null && recentMean !== null) {
    const support = supportAgainstBaseline(recent, baselineMean);
    direction = directionFromSupport(baselineMean, recentMean, support);
    magnitude = magnitudeFor(direction, recentMean - baselineMean);
  }

  return {
    policyVersion: BASELINE_POLICY_V1.version,
    dimension: input.dimension,
    dataState,
    direction,
    magnitude,
    confidence,
    baselineSamples: baseline.length,
    recentSamples: recent.length,
    baselineWindow:
      baseline.length > 0
        ? {
            from: new Date(baselineStartMs).toISOString(),
            to: new Date(recentStartMs).toISOString(),
          }
        : null,
    recentWindow:
      recent.length > 0
        ? { from: new Date(recentStartMs).toISOString(), to: new Date(nowMs).toISOString() }
        : null,
    sources: sourceSummary(observations, recentStartMs),
    explanation: explanationFor(
      input.dimension,
      dataState,
      direction,
      magnitude,
      confidence,
      baseline.length,
      recent.length
    ),
  };
}

export function evaluateBaselineSummary(input: {
  observations: readonly NormalizedObservation[];
  now: string;
}): BaselineSummary {
  const dimensions = Object.fromEntries(
    SIGNAL_DIMENSIONS.map((dimension) => [
      dimension,
      evaluateBaselineDimension({ dimension, observations: input.observations, now: input.now }),
    ])
  ) as Record<SignalDimension, BaselineDimension>;

  return {
    policyVersion: BASELINE_POLICY_V1.version,
    dimensions,
  };
}
