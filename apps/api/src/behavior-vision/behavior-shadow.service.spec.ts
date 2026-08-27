import { BehaviorShadowService } from './behavior-shadow.service';
import { BehaviorVisionService } from './behavior-vision.service';
import {
  BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorVisionReleaseQualification,
  type StoredBehaviorObservation,
} from './behavior-vision.types';

const ARTIFACT_SHA256 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const OLD_ARTIFACT_SHA256 = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';

const ACTIVE_RELEASE: BehaviorVisionReleaseQualification = {
  qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  qualified: true,
  releaseId: 'behavior-shadow-2026-08-27',
  modelVersion: 'shadow-model-1',
  featureVersion: 'features-1',
  artifactSha256: ARTIFACT_SHA256,
  responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
};

function observation(input: {
  id: string;
  context?: 'street' | 'park' | 'home';
  sessionKey?: string;
  phase?: 'baseline' | 'during-intervention' | 'recovery';
  accurate?: boolean;
  usable?: boolean;
  qualified?: boolean;
  release?: 'active' | 'old';
  startMs?: number;
  endMs?: number;
}): StoredBehaviorObservation {
  const qualified = input.qualified !== false;
  const release =
    input.release === 'old'
      ? {
          ...ACTIVE_RELEASE,
          releaseId: 'behavior-shadow-2026-07-01',
          modelVersion: 'shadow-model-0',
          featureVersion: 'features-0',
          artifactSha256: OLD_ARTIFACT_SHA256,
        }
      : ACTIVE_RELEASE;
  return {
    id: input.id,
    petId: 'pet-1',
    createdAt: '2026-08-23T12:00:00.000Z',
    mediaType: 'video',
    mediaSha256: `sha-${input.id}`,
    context: {
      context: input.context ?? 'street',
      sessionKey: input.sessionKey,
      phase: input.phase ?? 'baseline',
      handlerAction: input.phase === 'baseline' || !input.phase ? 'none' : 'increase-distance',
      leashState: 'loose',
      otherDogsPresent: true,
      audioAnalysisAllowed: false,
    },
    analysis: {
      schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
      modelVersion: release.modelVersion,
      featureVersion: release.featureVersion,
      ...(qualified
        ? {
            releaseId: release.releaseId,
            artifactSha256: release.artifactSha256,
            releaseQualification: release,
          }
        : {}),
      mediaQuality: {
        usable: input.usable ?? true,
        confidence: 0.9,
        issues: [],
        recaptureInstructions: [],
      },
      evidence: [
        {
          label: 'oriented toward dog',
          source: 'pose',
          confidence: 0.8,
          startMs: input.startMs ?? 1000,
          endMs: input.endMs ?? 1800,
        },
        {
          label: 'forward movement',
          source: 'motion',
          confidence: 0.9,
          startMs: (input.startMs ?? 1000) + 500,
          endMs: (input.endMs ?? 1800) + 700,
        },
      ],
      dimensions: [],
      hypotheses: [],
      observableSummary: 'Visible movement only.',
      uncertainty: 'Internal state cannot be determined.',
    },
    ...(input.accurate === undefined ? {} : { ownerFeedback: { accurate: input.accurate } }),
  };
}

function sameRelease(
  observation: StoredBehaviorObservation,
  active: BehaviorVisionReleaseQualification
) {
  const observed = observation.analysis.releaseQualification;
  return (
    observed?.qualified === true &&
    observed.qualificationVersion === active.qualificationVersion &&
    observed.releaseId === active.releaseId &&
    observed.modelVersion === active.modelVersion &&
    observed.featureVersion === active.featureVersion &&
    observed.artifactSha256 === active.artifactSha256 &&
    observed.responseContract === active.responseContract
  );
}

function makeVision(observations: StoredBehaviorObservation[]) {
  const active = observations.filter((entry) => sameRelease(entry, ACTIVE_RELEASE));
  return {
    timeline: jest.fn().mockResolvedValue(observations),
    activeReleaseQualification: jest.fn().mockReturnValue(ACTIVE_RELEASE),
    profile: jest.fn().mockResolvedValue({
      schemaVersion: 'woof-individual-behavior-profile-v1',
      petId: 'pet-1',
      sampleCount: active.length,
      contextsSeen: [...new Set(active.map((entry) => entry.context.context))],
      baselines: [],
      interventionEffects: [],
      personalizationConfidence: active.length ? 0.8 : 0,
      recommendation: {
        headline: 'Keep testing',
        explanation: 'Evidence only',
        nextSafeExperiment: [],
        neverAutoRecommendGreeting: true,
      },
    }),
  };
}

describe('BehaviorShadowService', () => {
  it('groups nearby timestamped evidence from the active release into reviewable moments', async () => {
    const vision = makeVision([observation({ id: 'obs-1', accurate: true })]);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.policy.mode).toBe('shadow-evidence-only');
    expect(result.policy.requiresQualifiedModelRelease).toBe(true);
    expect(result.policy.learningScope).toBe('active-qualified-release-only');
    expect(result.moments).toEqual([
      expect.objectContaining({
        observationId: 'obs-1',
        startMs: 1000,
        endMs: 2500,
        confidence: 0.9,
        labels: ['oriented toward dog', 'forward movement'],
        sources: ['pose', 'motion'],
      }),
    ]);
  });

  it('does not let confirmations on unusable observations inflate agreement', async () => {
    const observations = [observation({ id: 'usable-rejected', accurate: false })];
    for (let index = 0; index < 9; index += 1) {
      observations.push(
        observation({ id: `unusable-confirmed-${index}`, accurate: true, usable: false })
      );
    }
    const vision = makeVision(observations);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.evaluation.ownerReviewedObservations).toBe(1);
    expect(result.evaluation.ownerConfirmedObservations).toBe(0);
    expect(result.evaluation.ownerRejectedObservations).toBe(1);
    expect(result.evaluation.confirmationRate).toBe(0);
    expect(result.evaluation.evidenceReady).toBe(false);
  });

  it('separates legacy unqualified observations from active release evidence', async () => {
    const observations = [
      observation({ id: 'active', accurate: true }),
      observation({ id: 'legacy', accurate: true, qualified: false }),
    ];
    const vision = makeVision(observations);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.evaluation.observations).toBe(2);
    expect(result.evaluation.qualifiedObservations).toBe(1);
    expect(result.evaluation.activeReleaseObservations).toBe(1);
    expect(result.evaluation.inactiveQualifiedObservations).toBe(0);
    expect(result.evaluation.unqualifiedObservations).toBe(1);
    expect(result.evaluation.activeReleaseId).toBe(ACTIVE_RELEASE.releaseId);
    expect(result.evaluation.qualifiedReleaseIds).toEqual([ACTIVE_RELEASE.releaseId]);
    expect(result.moments.map((entry) => entry.observationId)).toEqual(['active']);
  });

  it('does not mix an older qualified release into current readiness or moments', async () => {
    const observations = [
      observation({ id: 'active', accurate: true }),
      observation({ id: 'old-qualified', accurate: true, release: 'old' }),
    ];
    const vision = makeVision(observations);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.evaluation.qualifiedObservations).toBe(2);
    expect(result.evaluation.activeReleaseObservations).toBe(1);
    expect(result.evaluation.inactiveQualifiedObservations).toBe(1);
    expect(result.evaluation.unqualifiedObservations).toBe(0);
    expect(result.evaluation.qualifiedReleaseIds).toEqual([
      'behavior-shadow-2026-07-01',
      'behavior-shadow-2026-08-27',
    ]);
    expect(result.evaluation.ownerReviewedObservations).toBe(1);
    expect(result.moments.map((entry) => entry.observationId)).toEqual(['active']);
  });

  it('never grants downstream authority even when active-release readiness gates are satisfied', async () => {
    const observations: StoredBehaviorObservation[] = [];
    for (let index = 0; index < 10; index += 1) {
      const sessionKey = `session-${index}`;
      const context = index % 3 === 0 ? 'home' : index % 3 === 1 ? 'park' : 'street';
      observations.push(
        observation({
          id: `base-${index}`,
          sessionKey,
          phase: 'baseline',
          context,
          accurate: true,
        }),
        observation({
          id: `recovery-${index}`,
          sessionKey,
          phase: 'recovery',
          context,
          accurate: true,
        })
      );
    }
    const vision = makeVision(observations);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.evaluation.evidenceReady).toBe(true);
    expect(result.evaluation.qualifiedObservations).toBe(20);
    expect(result.evaluation.activeReleaseObservations).toBe(20);
    expect(result.evaluation.usableObservations).toBe(20);
    expect(result.evaluation.ownerReviewedObservations).toBe(20);
    expect(result.evaluation.confirmationRate).toBe(1);
    expect(result.evaluation.pairedSessions).toBe(10);
    expect(result.policy).toEqual(
      expect.objectContaining({
        canInfluenceCompatibility: false,
        canMutateCanonicalPetState: false,
        canMakeSafetyDecision: false,
        promotionEnabled: false,
        promotionRequiresSeparateQualifiedRelease: true,
        requiresQualifiedModelRelease: true,
        learningScope: 'active-qualified-release-only',
      })
    );
  });
});
