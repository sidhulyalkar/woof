import { PrismaService } from '../prisma/prisma.service';
import { BehaviorVisionModelService } from './behavior-vision.model';
import { BehaviorVisionService } from './behavior-vision.service';

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
      analyze: jest.fn(),
    };
    const service = new BehaviorVisionService(
      prisma as unknown as PrismaService,
      model as unknown as BehaviorVisionModelService
    );

    const result = await service.analyze('user-1', dto, media);

    expect(result.provenance.pathway).toBe('model-unavailable');
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
      analyze: jest.fn().mockResolvedValue({
        schemaVersion: 'woof-behavior-observation-v1',
        modelVersion: 'shadow-test',
        featureVersion: 'features-test',
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
      }),
    };
    const service = new BehaviorVisionService(
      prisma as unknown as PrismaService,
      model as unknown as BehaviorVisionModelService
    );

    const result = await service.analyze('user-1', dto, media);

    expect(result.profile.recommendation.neverAutoRecommendGreeting).toBe(true);
    expect(result.coach.socialSafety).toContain('does not infer “needs to greet”');
    expect(model.analyze).toHaveBeenCalledWith(
      expect.objectContaining({
        context: expect.objectContaining({ audioAnalysisAllowed: false }),
      })
    );
  });
});