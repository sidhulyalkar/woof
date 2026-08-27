import { PrismaService } from '../prisma/prisma.service';
import { BehaviorVisionModelService } from './behavior-vision.model';
import { BehaviorVisionService } from './behavior-vision.service';
import {
  BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
  type BehaviorVisionReleaseQualification,
} from './behavior-vision.types';

const ARTIFACT_SHA256 = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const ACTIVE_RELEASE: BehaviorVisionReleaseQualification = {
  qualificationVersion: BEHAVIOR_MODEL_RELEASE_QUALIFICATION_VERSION,
  qualified: true,
  releaseId: 'behavior-shadow-2026-08-27',
  modelVersion: 'shadow-test',
  featureVersion: 'features-test',
  artifactSha256: ARTIFACT_SHA256,
  responseContract: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
};

function makePrisma() {
  return {
    pet: {
      findFirst: jest.fn().mockResolvedValue({
        id: 'pet-1',
        name: 'Nova',
        species: 'DOG',
        breed: 'Mixed',
        birthdate: null,
        temperament: null,
      }),
    },
    telemetry: {
      findMany: jest.fn().mockResolvedValue([]),
      findFirst: jest.fn(),
      create: jest.fn().mockResolvedValue({
        id: 'observation-1',
        createdAt: new Date('2026-08-20T12:00:00.000Z'),
      }),
      deleteMany: jest.fn(),
      delete: jest.fn(),
    },
    $transaction: jest.fn(),
  };
}

function qualifiedAnalysis() {
  return {
    schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
    releaseId: ACTIVE_RELEASE.releaseId,
    modelVersion: ACTIVE_RELEASE.modelVersion,
    featureVersion: ACTIVE_RELEASE.featureVersion,
    artifactSha256: ACTIVE_RELEASE.artifactSha256,
    releaseQualification: ACTIVE_RELEASE,
    mediaQuality: { usable: true, confidence: 0.9, issues: [], recaptureInstructions: [] },
    evidence: [],
    dimensions: [
      { dimension: 'arousal', value: 0.82, confidence: 0.9, basis: ['motion'] },
      {
        dimension: 'social-orientation',
        value: 0.8,
        confidence: 0.9,
        basis: ['orientation'],
      },
      {
        dimension: 'approach-tendency',
        value: 0.75,
        confidence: 0.9,
        basis: ['movement'],
      },
    ],
    hypotheses: [
      {
        id: 'barrier-frustration-compatible-pattern',
        confidence: 0.6,
        statement: 'Observed behavior is compatible with a high-arousal barrier pattern.',
        supportingEvidence: ['orientation'],
        contradictoryEvidence: [],
      },
    ],
    observableSummary: 'Nova oriented toward the other dog and repeatedly moved forward.',
    uncertainty: 'Internal motivation cannot be determined from this clip alone.',
  };
}

const dto = {
  petId: '00000000-0000-4000-8000-000000000001',
  context: 'street' as const,
  otherDogsPresent: true,
  leashState: 'loose' as const,
  phase: 'baseline' as const,
  handlerAction: 'none' as const,
  includeAudio: false,
  ownerNote: 'Another dog crossed the street.',
  saveToTimeline: true,
};

const media = {
  buffer: Buffer.from('private-video-bytes-that-must-not-be-persisted'),
  mimetype: 'video/webm',
  originalname: 'nova.webm',
} as Express.Multer.File;

describe('BehaviorVisionService', () => {
  it('fails closed when the specialized model is unavailable and never persists raw media bytes', async () => {
    const prisma = makePrisma();
    const model = {
      isConfigured: jest.fn().mockReturnValue(false),
      activeReleaseQualification: jest.fn().mockReturnValue(null),
      analyze: jest.fn(),
    };
    const service = new BehaviorVisionService(
      prisma as unknown as PrismaService,
      model as unknown as BehaviorVisionModelService
    );

    const result = await service.analyze('user-1', dto, media);

    expect(result.provenance.pathway).toBe('model-unavailable');
    expect(result.provenance.modelReleaseQualified).toBe(false);
    expect(result.provenance.activeReleaseId).toBeNull();
    expect(result.analysis.mediaQuality.usable).toBe(false);
    expect(result.analysis.dimensions).toEqual([]);
    expect(model.analyze).not.toHaveBeenCalled();
    expect(result.privacy.mediaStoredByWoof).toBe(false);
    expect(result.privacy.audioAnalysisAllowed).toBe(false);

    const createCall = prisma.telemetry.create.mock.calls[0][0];
    const persisted = JSON.stringify(createCall);
    expect(persisted).not.toContain('private-video-bytes-that-must-not-be-persisted');
    expect(persisted).toContain('mediaSha256');
    expect(persisted).toContain('"audioAnalysisAllowed":false');
  });

  it('keeps direct greeting out of the automated coaching recommendation', async () => {
    const prisma = makePrisma();
    const model = {
      isConfigured: jest.fn().mockReturnValue(true),
      activeReleaseQualification: jest.fn().mockReturnValue(ACTIVE_RELEASE),
      analyze: jest.fn().mockResolvedValue(qualifiedAnalysis()),
    };
    const service = new BehaviorVisionService(
      prisma as unknown as PrismaService,
      model as unknown as BehaviorVisionModelService
    );

    const result = await service.analyze('user-1', dto, media);

    expect(result.profile.recommendation.neverAutoRecommendGreeting).toBe(true);
    expect(result.provenance.modelReleaseQualified).toBe(true);
    expect(result.provenance.activeReleaseId).toBe(ACTIVE_RELEASE.releaseId);
    expect(result.coach.socialSafety).toContain('does not infer “needs to greet”');
    expect(model.analyze).toHaveBeenCalledWith(
      expect.objectContaining({
        context: expect.objectContaining({ audioAnalysisAllowed: false }),
      })
    );
  });

  it('conditions the current model only on observations from the active release', async () => {
    const prisma = makePrisma();
    const oldRelease = {
      ...ACTIVE_RELEASE,
      releaseId: 'behavior-shadow-2026-07-01',
      modelVersion: 'shadow-test-old',
      artifactSha256: 'b'.repeat(64),
    };
    const observationData = (release: BehaviorVisionReleaseQualification, arousal: number) => ({
      schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
      mediaType: 'video',
      mediaSha256: `sha-${release.releaseId}`,
      context: {
        context: 'street',
        phase: 'baseline',
        handlerAction: 'none',
        leashState: 'loose',
        otherDogsPresent: true,
        audioAnalysisAllowed: false,
      },
      analysis: {
        schemaVersion: BEHAVIOR_OBSERVATION_SCHEMA_VERSION,
        releaseId: release.releaseId,
        modelVersion: release.modelVersion,
        featureVersion: release.featureVersion,
        artifactSha256: release.artifactSha256,
        releaseQualification: release,
        mediaQuality: { usable: true, confidence: 0.9, issues: [], recaptureInstructions: [] },
        evidence: [],
        dimensions: [{ dimension: 'arousal', value: arousal, confidence: 0.9, basis: ['fixture'] }],
        hypotheses: [],
        observableSummary: 'fixture',
        uncertainty: 'fixture',
      },
    });
    prisma.telemetry.findMany.mockImplementation(async (args: { where?: { event?: string } }) => {
      if (args.where?.event === 'BEHAVIOR_OBSERVATION_FEEDBACK') return [];
      return [
        {
          id: 'active-observation',
          petId: 'pet-1',
          createdAt: new Date('2026-08-25T12:00:00.000Z'),
          data: observationData(ACTIVE_RELEASE, 0.2),
        },
        {
          id: 'old-observation',
          petId: 'pet-1',
          createdAt: new Date('2026-08-24T12:00:00.000Z'),
          data: observationData(oldRelease, 0.95),
        },
      ];
    });
    const model = {
      isConfigured: jest.fn().mockReturnValue(true),
      activeReleaseQualification: jest.fn().mockReturnValue(ACTIVE_RELEASE),
      analyze: jest.fn().mockResolvedValue(qualifiedAnalysis()),
    };
    const service = new BehaviorVisionService(
      prisma as unknown as PrismaService,
      model as unknown as BehaviorVisionModelService
    );

    await service.analyze('user-1', dto, media);

    expect(model.analyze).toHaveBeenCalledWith(
      expect.objectContaining({
        priorProfileSummary: expect.objectContaining({
          sampleCount: 1,
          baselines: expect.arrayContaining([
            expect.objectContaining({ dimension: 'arousal', mean: expect.closeTo(0.2, 4) }),
          ]),
        }),
      })
    );
  });
});
