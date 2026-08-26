import type {
  BaselineDimension,
  EvidenceReliability,
  EvidenceSourceType,
  NormalizedObservation,
  SignalDimension,
} from './baseline-policy-v1.types';

export const BASELINE_FIXTURE_NOW = '2026-08-25T12:00:00.000Z';
const DAY_MS = 86_400_000;

type ObservationInput = {
  id: string;
  daysAgo: number;
  delta: -2 | -1 | 0 | 1 | 2;
  dimension?: SignalDimension;
  sourceType?: EvidenceSourceType;
  reliability?: EvidenceReliability;
  confidence?: number;
  dedupeKey?: string;
  localDate?: string;
  supersedesObservationId?: string;
};

export function fixtureObservation(input: ObservationInput): NormalizedObservation {
  const observedAt = new Date(
    Date.parse(BASELINE_FIXTURE_NOW) - input.daysAgo * DAY_MS
  ).toISOString();
  return {
    id: input.id,
    dedupeKey: input.dedupeKey ?? input.id,
    dimension: input.dimension ?? 'APPETITE',
    observedAt,
    localDate: input.localDate ?? observedAt.slice(0, 10),
    deltaBucket: input.delta,
    sourceType: input.sourceType ?? 'OWNER_CHECKIN',
    reliability: input.reliability ?? 'STANDARD',
    confidence: input.confidence ?? 1,
    ...(input.supersedesObservationId
      ? { supersedesObservationId: input.supersedesObservationId }
      : {}),
  };
}

function baselineZeros(count: number, startDaysAgo = count + 3): NormalizedObservation[] {
  return Array.from({ length: count }, (_, index) =>
    fixtureObservation({
      id: `baseline-${startDaysAgo - index}`,
      daysAgo: startDaysAgo - index,
      delta: 0,
    })
  );
}

function stableBaseline(): NormalizedObservation[] {
  return [
    ...baselineZeros(7, 10),
    fixtureObservation({ id: 'recent-stable-1', daysAgo: 1, delta: 0 }),
    fixtureObservation({ id: 'recent-stable-0', daysAgo: 0, delta: 0 }),
  ];
}

type ExpectedFixtureReceipt = Pick<
  BaselineDimension,
  'dataState' | 'direction' | 'magnitude' | 'confidence' | 'baselineSamples' | 'recentSamples'
>;

export type BaselinePolicyFixture = {
  name: string;
  dimension: SignalDimension;
  observations: readonly NormalizedObservation[];
  expected: ExpectedFixtureReceipt;
};

export const BASELINE_POLICY_FIXTURES: readonly BaselinePolicyFixture[] = [
  {
    name: 'zero evidence',
    dimension: 'APPETITE',
    observations: [],
    expected: {
      dataState: 'INSUFFICIENT',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'LOW',
      baselineSamples: 0,
      recentSamples: 0,
    },
  },
  {
    name: 'one owner sample',
    dimension: 'APPETITE',
    observations: [fixtureObservation({ id: 'one', daysAgo: 1, delta: 0 })],
    expected: {
      dataState: 'INSUFFICIENT',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'LOW',
      baselineSamples: 0,
      recentSamples: 1,
    },
  },
  {
    name: 'learning threshold minus one',
    dimension: 'APPETITE',
    observations: [fixtureObservation({ id: 'learn-1', daysAgo: 4, delta: 0 })],
    expected: {
      dataState: 'INSUFFICIENT',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'LOW',
      baselineSamples: 1,
      recentSamples: 0,
    },
  },
  {
    name: 'learning threshold exactly reached',
    dimension: 'APPETITE',
    observations: [
      fixtureObservation({ id: 'learn-2a', daysAgo: 5, delta: 0 }),
      fixtureObservation({ id: 'learn-2b', daysAgo: 4, delta: 0 }),
    ],
    expected: {
      dataState: 'LEARNING',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'LOW',
      baselineSamples: 2,
      recentSamples: 0,
    },
  },
  {
    name: 'stable baseline',
    dimension: 'APPETITE',
    observations: stableBaseline(),
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'irrelevant dimensions cannot alter stable baseline',
    dimension: 'APPETITE',
    observations: [
      ...stableBaseline(),
      fixtureObservation({ id: 'energy-noise-1', daysAgo: 1, delta: -2, dimension: 'ENERGY' }),
      fixtureObservation({ id: 'energy-noise-2', daysAgo: 0, delta: 2, dimension: 'ENERGY' }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'gradual sustained lower shift',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      fixtureObservation({ id: 'lower-2', daysAgo: 2, delta: -1 }),
      fixtureObservation({ id: 'lower-1', daysAgo: 1, delta: -1 }),
      fixtureObservation({ id: 'lower-0', daysAgo: 0, delta: -1 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'LOWER',
      magnitude: 'MODERATE',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 3,
    },
  },
  {
    name: 'gradual sustained higher shift',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      fixtureObservation({ id: 'higher-2', daysAgo: 2, delta: 1 }),
      fixtureObservation({ id: 'higher-1', daysAgo: 1, delta: 1 }),
      fixtureObservation({ id: 'higher-0', daysAgo: 0, delta: 1 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'HIGHER',
      magnitude: 'MODERATE',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 3,
    },
  },
  {
    name: 'single extreme outlier then normal',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      fixtureObservation({ id: 'outlier', daysAgo: 1, delta: -2 }),
      fixtureObservation({ id: 'outlier-normal', daysAgo: 0, delta: 0 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'repeated large shift',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(10, 13),
      fixtureObservation({ id: 'large-2', daysAgo: 2, delta: -2 }),
      fixtureObservation({ id: 'large-1', daysAgo: 1, delta: -2 }),
      fixtureObservation({ id: 'large-0', daysAgo: 0, delta: -2 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'LOWER',
      magnitude: 'LARGE',
      confidence: 'HIGH',
      baselineSamples: 10,
      recentSamples: 3,
    },
  },
  {
    name: 'conflicting owner and activity evidence',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(10, 13),
      fixtureObservation({
        id: 'conflict-owner',
        daysAgo: 2,
        delta: -1,
        sourceType: 'OWNER_CHECKIN',
        reliability: 'STRONG',
      }),
      fixtureObservation({
        id: 'conflict-activity',
        daysAgo: 1,
        delta: 1,
        sourceType: 'ACTIVITY',
      }),
      fixtureObservation({
        id: 'conflict-coach',
        daysAgo: 0,
        delta: -1,
        sourceType: 'COACHING',
      }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'MIXED',
      magnitude: 'UNAVAILABLE',
      confidence: 'MEDIUM',
      baselineSamples: 10,
      recentSamples: 3,
    },
  },
  {
    name: 'stale baseline',
    dimension: 'APPETITE',
    observations: Array.from({ length: 7 }, (_, index) =>
      fixtureObservation({ id: `stale-${14 - index}`, daysAgo: 14 - index, delta: 0 })
    ),
    expected: {
      dataState: 'STALE',
      direction: 'UNAVAILABLE',
      magnitude: 'UNAVAILABLE',
      confidence: 'LOW',
      baselineSamples: 7,
      recentSamples: 0,
    },
  },
  {
    name: 'recovery toward baseline',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(10, 13),
      fixtureObservation({ id: 'recovery-2', daysAgo: 2, delta: -1 }),
      fixtureObservation({ id: 'recovery-1', daysAgo: 1, delta: 0 }),
      fixtureObservation({ id: 'recovery-0', daysAgo: 0, delta: 0 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 10,
      recentSamples: 3,
    },
  },
  {
    name: 'late arriving historical evidence uses observed time',
    dimension: 'APPETITE',
    observations: [...stableBaseline()].reverse(),
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'future timestamp is excluded',
    dimension: 'APPETITE',
    observations: [
      ...stableBaseline(),
      fixtureObservation({ id: 'future', daysAgo: -1, delta: -2, reliability: 'STRONG' }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'duplicate normalized evidence does not add authority',
    dimension: 'APPETITE',
    observations: [
      ...stableBaseline(),
      fixtureObservation({
        id: 'duplicate-copy',
        dedupeKey: 'recent-stable-0',
        daysAgo: 0,
        delta: 0,
      }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'correction supersedes original evidence',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      fixtureObservation({ id: 'mistaken-report', daysAgo: 1, delta: -2 }),
      fixtureObservation({
        id: 'corrected-report',
        daysAgo: 1,
        delta: 0,
        supersedesObservationId: 'mistaken-report',
      }),
      fixtureObservation({ id: 'correction-normal', daysAgo: 0, delta: 0 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'sparse alternating evidence stays mixed',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      fixtureObservation({ id: 'alternating-low', daysAgo: 2, delta: -1 }),
      fixtureObservation({ id: 'alternating-high', daysAgo: 0, delta: 1 }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'MIXED',
      magnitude: 'UNAVAILABLE',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'history outside bounded policy window is irrelevant',
    dimension: 'APPETITE',
    observations: [
      ...stableBaseline(),
      fixtureObservation({ id: 'ancient', daysAgo: 40, delta: -2, reliability: 'STRONG' }),
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
  {
    name: 'timezone boundary uses upstream normalized local date',
    dimension: 'APPETITE',
    observations: [
      ...baselineZeros(7, 10),
      {
        ...fixtureObservation({ id: 'tz-recent-1', daysAgo: 1, delta: 0 }),
        observedAt: '2026-08-24T07:30:00.000Z',
        localDate: '2026-08-23',
      },
      {
        ...fixtureObservation({ id: 'tz-recent-0', daysAgo: 0, delta: 0 }),
        observedAt: '2026-08-25T07:30:00.000Z',
        localDate: '2026-08-24',
      },
    ],
    expected: {
      dataState: 'ESTABLISHED',
      direction: 'NEAR_BASELINE',
      magnitude: 'SMALL',
      confidence: 'MEDIUM',
      baselineSamples: 7,
      recentSamples: 2,
    },
  },
];
