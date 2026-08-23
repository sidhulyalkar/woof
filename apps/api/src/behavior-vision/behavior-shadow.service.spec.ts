import { BehaviorShadowService } from './behavior-shadow.service';
import { BehaviorVisionService } from './behavior-vision.service';
import type { StoredBehaviorObservation } from './behavior-vision.types';

function observation(input: {
  id: string;
  context?: 'street' | 'park' | 'home';
  sessionKey?: string;
  phase?: 'baseline' | 'during-intervention' | 'recovery';
  accurate?: boolean;
  startMs?: number;
  endMs?: number;
}): StoredBehaviorObservation {
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
      schemaVersion: 'woof-behavior-observation-v1',
      modelVersion: 'shadow-model-1',
      featureVersion: 'features-1',
      mediaQuality: { usable: true, confidence: 0.9, issues: [], recaptureInstructions: [] },
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

function makeVision(observations: StoredBehaviorObservation[]) {
  return {
    timeline: jest.fn().mockResolvedValue(observations),
    profile: jest.fn().mockResolvedValue({
      schemaVersion: 'woof-individual-behavior-profile-v1',
      petId: 'pet-1',
      sampleCount: observations.length,
      contextsSeen: ['home', 'park', 'street'],
      baselines: [],
      interventionEffects: [],
      personalizationConfidence: 0.8,
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
  it('groups nearby timestamped evidence into reviewable moments', async () => {
    const vision = makeVision([observation({ id: 'obs-1', accurate: true })]);
    const service = new BehaviorShadowService(vision as unknown as BehaviorVisionService);

    const result = await service.snapshot('user-1', 'pet-1');

    expect(result.policy.mode).toBe('shadow-evidence-only');
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

  it('never grants downstream authority even when evidence readiness gates are satisfied', async () => {
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
      })
    );
  });
});
