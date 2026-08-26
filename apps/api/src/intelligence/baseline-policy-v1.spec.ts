import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import {
  BASELINE_POLICY_FIXTURES,
  BASELINE_FIXTURE_NOW,
  fixtureObservation,
} from './baseline-policy-v1.fixtures';
import { BASELINE_POLICY_V1 } from './baseline-policy-v1.receipt';
import { evaluateBaselineDimension, evaluateBaselineSummary } from './baseline-policy-v1';
import {
  SIGNAL_DIMENSIONS,
  type Confidence,
  type DeltaBucket,
  type NormalizedObservation,
} from './baseline-policy-v1.types';

const CONFIDENCE_RANK: Record<Confidence, number> = {
  LOW: 0,
  MEDIUM: 1,
  HIGH: 2,
};

function establishedEvidence(): NormalizedObservation[] {
  return [
    ...Array.from({ length: 10 }, (_, index) =>
      fixtureObservation({ id: `property-baseline-${13 - index}`, daysAgo: 13 - index, delta: 0 })
    ),
    fixtureObservation({ id: 'property-recent-2', daysAgo: 2, delta: -2 }),
    fixtureObservation({ id: 'property-recent-1', daysAgo: 1, delta: -2 }),
    fixtureObservation({ id: 'property-recent-0', daysAgo: 0, delta: -2 }),
  ];
}

function evidenceForValues(
  baselineValues: readonly DeltaBucket[],
  recentValues: readonly DeltaBucket[]
): NormalizedObservation[] {
  return [
    ...baselineValues.map((delta, index) =>
      fixtureObservation({
        id: `matrix-baseline-${index}-${delta}`,
        daysAgo: 4 + index,
        delta,
      })
    ),
    ...recentValues.map((delta, index) =>
      fixtureObservation({
        id: `matrix-recent-${index}-${delta}`,
        daysAgo: index % 3,
        delta,
      })
    ),
  ];
}

function cartesianRecentValues(maxLength: number): DeltaBucket[][] {
  const values: DeltaBucket[] = [-2, -1, 0, 1, 2];
  const results: DeltaBucket[][] = [];

  function visit(prefix: DeltaBucket[], targetLength: number) {
    if (prefix.length === targetLength) {
      results.push(prefix);
      return;
    }
    for (const value of values) visit([...prefix, value], targetLength);
  }

  for (let length = 1; length <= maxLength; length += 1) visit([], length);
  return results;
}

function evaluate(observations: readonly NormalizedObservation[]) {
  return evaluateBaselineDimension({
    dimension: 'APPETITE',
    observations,
    now: BASELINE_FIXTURE_NOW,
  });
}

describe('baseline-policy-v1 fixture oracle', () => {
  it.each(BASELINE_POLICY_FIXTURES)('$name', (fixture) => {
    const result = evaluateBaselineDimension({
      dimension: fixture.dimension,
      observations: fixture.observations,
      now: BASELINE_FIXTURE_NOW,
    });

    expect(result).toMatchObject(fixture.expected);
    expect(result.policyVersion).toBe('baseline-policy-v1');
    expect(result.explanation.length).toBeGreaterThan(0);
  });

  it('pins the complete policy receipt under the v1 identity', () => {
    expect(BASELINE_POLICY_V1).toEqual({
      version: 'baseline-policy-v1',
      retentionWindowDays: 31,
      baselineWindowDays: 28,
      recentWindowDays: 3,
      learningMinDistinctDays: 2,
      establishedMinDistinctDays: 7,
      staleAfterDays: 7,
      minimumDirectionalSamples: 2,
      directionThreshold: 0.35,
      magnitudeThresholds: { moderate: 0.8, large: 1.35 },
      conflictRatio: 0.35,
      reliabilityWeights: { WEAK: 0.5, STANDARD: 1, STRONG: 1.5 },
      confidenceAgreementBonusPerSample: 1,
      confidenceThresholds: {
        mediumBaselineWeight: 5,
        mediumRecentEvidenceScore: 3,
        highBaselineWeight: 9,
        highRecentEvidenceScore: 6,
      },
    });

    const fingerprint = createHash('sha256')
      .update(JSON.stringify(BASELINE_POLICY_V1))
      .digest('hex');
    expect(fingerprint).toBe('1d8b9432f3ac3bb18afdbea7b7defe3032daecebd0bf6eac84d3c2e2190ac4c3');
  });
});

describe('baseline-policy-v1 metamorphic contracts', () => {
  it('is permutation invariant for canonically identifiable evidence', () => {
    const evidence = establishedEvidence();
    const forward = evaluate(evidence);
    const reverse = evaluate([...evidence].reverse());
    const rotated = evaluate([...evidence.slice(5), ...evidence.slice(0, 5)]);

    expect(reverse).toEqual(forward);
    expect(rotated).toEqual(forward);
  });

  it('is byte-equivalent for identical explicit inputs', () => {
    const evidence = establishedEvidence();
    const first = JSON.stringify(evaluate(evidence));
    const second = JSON.stringify(evaluate(evidence));
    expect(second).toBe(first);
  });

  it('ignores evidence from unrelated dimensions', () => {
    const evidence = establishedEvidence();
    const baseline = evaluate(evidence);
    const noisy = evaluate([
      ...evidence,
      fixtureObservation({ id: 'noise-energy', daysAgo: 0, delta: 2, dimension: 'ENERGY' }),
      fixtureObservation({
        id: 'noise-mobility',
        daysAgo: 1,
        delta: -2,
        dimension: 'MOBILITY_COMFORT',
      }),
    ]);
    expect(noisy).toEqual(baseline);
  });

  it('ignores evidence outside the bounded policy window', () => {
    const evidence = establishedEvidence();
    expect(
      evaluate([
        ...evidence,
        fixtureObservation({
          id: 'too-old-to-matter',
          daysAgo: BASELINE_POLICY_V1.retentionWindowDays + 10,
          delta: 2,
          reliability: 'STRONG',
        }),
      ])
    ).toEqual(evaluate(evidence));
  });

  it('cannot gain confidence when effective evidence is removed', () => {
    const evidence = establishedEvidence();
    const original = evaluate(evidence);

    for (let index = 0; index < evidence.length; index += 1) {
      const reduced = evaluate(evidence.filter((_, candidate) => candidate !== index));
      expect(CONFIDENCE_RANK[reduced.confidence]).toBeLessThanOrEqual(
        CONFIDENCE_RANK[original.confidence]
      );
    }
  });

  it('keeps the historical confidence-removal counterexample monotone', () => {
    const evidence = evidenceForValues([1, -2, 1, -1, 2, -2, -1, -2, 1], [1, 0, -2]);
    const original = evaluate(evidence);
    const reduced = evaluate(evidence.filter((_, index) => index !== 0));

    expect(original.confidence).toBe('MEDIUM');
    expect(reduced.confidence).toBe('MEDIUM');
    expect(CONFIDENCE_RANK[reduced.confidence]).toBeLessThanOrEqual(
      CONFIDENCE_RANK[original.confidence]
    );
  });

  it('keeps confidence monotone across an adversarial deletion matrix', () => {
    const baselineShapes: DeltaBucket[][] = [
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, -2, 1, -1, 2, -2, -1, -2, 1],
      [-2, -1, 0, 1, 2, -1, 1, 0, 2],
      [2, 2, 1, 1, 0, -1, -1, -2, -2, 0],
    ];
    const recentShapes = cartesianRecentValues(3);

    for (const baselineValues of baselineShapes) {
      for (const recentValues of recentShapes) {
        const evidence = evidenceForValues(baselineValues, recentValues);
        const original = evaluate(evidence);
        const originalRank = CONFIDENCE_RANK[original.confidence];

        for (let index = 0; index < evidence.length; index += 1) {
          const reduced = evaluate(evidence.filter((_, candidate) => candidate !== index));
          expect(CONFIDENCE_RANK[reduced.confidence]).toBeLessThanOrEqual(originalRank);
        }
      }
    }
  });

  it('gives agreement more confidence than the same amount of conflicting evidence', () => {
    const baseline: DeltaBucket[] = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const conflicting = evaluate(evidenceForValues(baseline, [-1, -1, 1]));
    const agreeing = evaluate(evidenceForValues(baseline, [-1, -1, -1]));

    expect(conflicting.direction).toBe('MIXED');
    expect(conflicting.confidence).toBe('MEDIUM');
    expect(agreeing.direction).toBe('LOWER');
    expect(agreeing.confidence).toBe('HIGH');
  });

  it('cannot gain confidence when a strong source is replaced by a weaker source', () => {
    const evidence = establishedEvidence();
    const strong = evidence.map((observation, index) =>
      index === evidence.length - 1
        ? { ...observation, reliability: 'STRONG' as const }
        : observation
    );
    const weak = strong.map((observation, index) =>
      index === strong.length - 1 ? { ...observation, reliability: 'WEAK' as const } : observation
    );

    expect(CONFIDENCE_RANK[evaluate(weak).confidence]).toBeLessThanOrEqual(
      CONFIDENCE_RANK[evaluate(strong).confidence]
    );
  });

  it('does not gain confidence from a duplicate dedupe key', () => {
    const evidence = establishedEvidence();
    const duplicate = {
      ...evidence.at(-1)!,
      id: 'duplicate-property-copy',
    };
    expect(evaluate([...evidence, duplicate])).toEqual(evaluate(evidence));
  });

  it('never turns missing observations into near-baseline reassurance', () => {
    const result = evaluate([]);
    expect(result.direction).toBe('UNAVAILABLE');
    expect(result.dataState).toBe('INSUFFICIENT');
  });

  it('cannot produce HIGH confidence and LARGE magnitude from one weak recent source', () => {
    const result = evaluate([
      ...establishedEvidence().slice(0, 10),
      fixtureObservation({ id: 'one-weak-shift', daysAgo: 0, delta: -2, reliability: 'WEAK' }),
    ]);
    expect(result.direction).toBe('UNAVAILABLE');
    expect(result.confidence).toBe('LOW');
    expect(result.magnitude).toBe('UNAVAILABLE');
  });

  it('requires a valid explicit now instead of reading a process clock', () => {
    expect(() =>
      evaluateBaselineDimension({ dimension: 'APPETITE', observations: [], now: 'not-a-timestamp' })
    ).toThrow('baseline-policy-v1 requires a valid explicit now timestamp');
  });

  it('never emits non-finite numeric output or an aggregate health score', () => {
    const summary = evaluateBaselineSummary({
      observations: establishedEvidence(),
      now: BASELINE_FIXTURE_NOW,
    });
    const encoded = JSON.stringify(summary);
    expect(encoded).not.toContain('NaN');
    expect(encoded).not.toContain('Infinity');
    expect(encoded.toLowerCase()).not.toContain('healthscore');
    expect(encoded.toLowerCase()).not.toContain('wellnessscore');
    expect(Object.keys(summary.dimensions).sort()).toEqual([...SIGNAL_DIMENSIONS].sort());
  });

  it('keeps explanation branches aligned with structured state', () => {
    for (const fixture of BASELINE_POLICY_FIXTURES) {
      const result = evaluateBaselineDimension({
        dimension: fixture.dimension,
        observations: fixture.observations,
        now: BASELINE_FIXTURE_NOW,
      });
      if (result.dataState === 'INSUFFICIENT')
        expect(result.explanation).toContain('Not enough valid');
      if (result.dataState === 'LEARNING') expect(result.explanation).toContain('still learning');
      if (result.dataState === 'STALE') expect(result.explanation).toContain('baseline is stale');
      if (result.direction === 'MIXED')
        expect(result.explanation).toContain('different directions');
      if (result.direction === 'NEAR_BASELINE')
        expect(result.explanation).toContain('near the established baseline');
      if (result.direction === 'LOWER') expect(result.explanation).toContain('lower than');
      if (result.direction === 'HIGHER') expect(result.explanation).toContain('higher than');
    }
  });
});

describe('baseline-policy-v1 bounded performance', () => {
  it('evaluates a realistic six-dimension summary within a generous deterministic CI budget', () => {
    const observations = SIGNAL_DIMENSIONS.flatMap((dimension, dimensionIndex) =>
      Array.from({ length: 31 }, (_, index) =>
        fixtureObservation({
          id: `perf-${dimension}-${index}`,
          daysAgo: 30 - index,
          delta: (((index + dimensionIndex) % 5) - 2) as -2 | -1 | 0 | 1 | 2,
          dimension,
          sourceType: index % 2 === 0 ? 'OWNER_CHECKIN' : 'ACTIVITY',
        })
      )
    );

    const start = performance.now();
    for (let iteration = 0; iteration < 100; iteration += 1) {
      evaluateBaselineSummary({ observations, now: BASELINE_FIXTURE_NOW });
    }
    const elapsedMs = performance.now() - start;

    expect(elapsedMs).toBeLessThan(2_000);
  });
});
