import { deriveIndividualBehaviorProfile } from './behavior-profile';
import {
  BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorDimension,
  type HandlerAction,
  type StoredBehaviorObservation,
} from './behavior-vision.types';

const ARTIFACT_SHA256 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';

function makeObservation(input: {
  id: string;
  createdAt: string;
  sessionKey?: string;
  phase?: 'baseline' | 'during-intervention' | 'recovery';
  action?: HandlerAction;
  context?: 'street' | 'park' | 'home';
  values?: Partial<Record<BehaviorDimension, number>>;
  accurate?: boolean;
  qualified?: boolean;
}): StoredBehaviorObservation {
  const values = input.values ?? {};
  const qualified = input.qualified !== false;
  return {
    id: input.id,
    petId: 'pet-1',
    createdAt: input.createdAt,
    mediaType: 'video',
    mediaSha256: `sha-${input.id}`,
    context: {
      context: input.context ?? 'street',
      sessionKey: input.sessionKey,
      phase: input.phase ?? 'baseline',
      handlerAction: input.action ?? 'none',
      leashState: 'loose',
      otherDogsPresent: true,
      audioAnalysisAllowed: false,
    },
    analysis: {
      schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
      modelVersion: 'test-model',
      featureVersion: 'test-features',
      ...(qualified
        ? {
            releaseId: 'test-release',
            artifactSha256: ARTIFACT_SHA256,
            releaseQualification: {
              qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
              qualified: true as const,
              releaseId: 'test-release',
              modelVersion: 'test-model',
              featureVersion: 'test-features',
              artifactSha256: ARTIFACT_SHA256,
              responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
            },
          }
        : {}),
      mediaQuality: {
        usable: true,
        confidence: 0.9,
        issues: [],
        recaptureInstructions: [],
      },
      evidence: [],
      dimensions: Object.entries(values).map(([dimension, value]) => ({
        dimension: dimension as BehaviorDimension,
        value: value ?? 0,
        confidence: 0.9,
        basis: ['test'],
      })),
      hypotheses: [],
      observableSummary: 'test',
      uncertainty: 'test fixture',
    },
    ownerFeedback:
      input.accurate === undefined
        ? undefined
        : {
            accurate: input.accurate,
          },
  };
}

describe('individual behavior profile', () => {
  it('starts conservatively and never auto-recommends a dog greeting', () => {
    const profile = deriveIndividualBehaviorProfile('pet-1', []);
    expect(profile.sampleCount).toBe(0);
    expect(profile.recommendation.headline).toContain('still learning');
    expect(profile.recommendation.neverAutoRecommendGreeting).toBe(true);
  });

  it('does not equate high social orientation and arousal with a need to greet', () => {
    const observations = Array.from({ length: 6 }, (_, index) =>
      makeObservation({
        id: `o-${index}`,
        createdAt: new Date(2026, 0, index + 1).toISOString(),
        context: index < 3 ? 'street' : 'park',
        values: {
          arousal: 0.82,
          'social-orientation': 0.84,
          'approach-tendency': 0.76,
          'avoidance-tendency': 0.22,
          'body-tension': 0.55,
        },
      })
    );

    const profile = deriveIndividualBehaviorProfile('pet-1', observations);
    expect(profile.recommendation.headline).toContain('does not automatically mean');
    expect(profile.recommendation.explanation).toContain('barrier frustration');
    expect(profile.recommendation.neverAutoRecommendGreeting).toBe(true);
  });

  it('learns a repeated within-dog handler strategy when paired observations improve', () => {
    const observations: StoredBehaviorObservation[] = [];
    for (let session = 0; session < 3; session += 1) {
      const day = session + 1;
      observations.push(
        makeObservation({
          id: `b-${session}`,
          createdAt: new Date(2026, 1, day, 10, 0).toISOString(),
          sessionKey: `session-${session}`,
          phase: 'baseline',
          values: {
            arousal: 0.82,
            'body-tension': 0.74,
            'handler-engagement': 0.28,
          },
        }),
        makeObservation({
          id: `r-${session}`,
          createdAt: new Date(2026, 1, day, 10, 1).toISOString(),
          sessionKey: `session-${session}`,
          phase: 'recovery',
          action: 'increase-distance',
          values: {
            arousal: 0.42,
            'body-tension': 0.34,
            'handler-engagement': 0.66,
          },
        })
      );
    }

    const profile = deriveIndividualBehaviorProfile('pet-1', observations);
    const effect = profile.interventionEffects.find(
      (entry) => entry.action === 'increase-distance'
    );
    expect(effect?.pairedSessions).toBe(3);
    expect(effect?.arousalDelta).toBeLessThan(-0.3);
    expect(effect?.tensionDelta).toBeLessThan(-0.3);
    expect(profile.recommendation.explanation).toContain('increase-distance');
  });

  it('does not let owner-rejected model observations shape the personal baseline', () => {
    const profile = deriveIndividualBehaviorProfile('pet-1', [
      makeObservation({
        id: 'bad',
        createdAt: new Date(2026, 2, 1).toISOString(),
        accurate: false,
        values: { arousal: 1, 'body-tension': 1 },
      }),
      makeObservation({
        id: 'good',
        createdAt: new Date(2026, 2, 2).toISOString(),
        accurate: true,
        values: { arousal: 0.2, 'body-tension': 0.25 },
      }),
    ]);

    const arousal = profile.baselines.find((entry) => entry.dimension === 'arousal');
    expect(arousal?.sampleCount).toBe(1);
    expect(arousal?.mean).toBeCloseTo(0.2);
  });

  it('does not let owner confirmation promote a legacy unqualified model observation', () => {
    const profile = deriveIndividualBehaviorProfile('pet-1', [
      makeObservation({
        id: 'legacy-confirmed',
        createdAt: new Date(2026, 2, 3).toISOString(),
        accurate: true,
        qualified: false,
        values: { arousal: 0.95, 'body-tension': 0.9 },
      }),
      makeObservation({
        id: 'qualified-current',
        createdAt: new Date(2026, 2, 4).toISOString(),
        accurate: true,
        values: { arousal: 0.2, 'body-tension': 0.25 },
      }),
    ]);

    expect(profile.sampleCount).toBe(1);
    const arousal = profile.baselines.find((entry) => entry.dimension === 'arousal');
    expect(arousal?.sampleCount).toBe(1);
    expect(arousal?.mean).toBeCloseTo(0.2);
  });
});
