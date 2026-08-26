import { ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { HealthAiService, type PetHealthModelResult } from './health-ai.service';
import { HealthLensService } from './health-lens.service';

const pet = {
  id: 'pet-1',
  name: 'Milo',
  species: 'DOG',
  breed: 'Mixed breed',
  birthdate: new Date('2022-01-01T00:00:00.000Z'),
  temperament: { friendly: 4 },
};

const safeModelResult: PetHealthModelResult = {
  triage: 'monitor',
  confidence: 0.72,
  summary: 'A localized skin change is visible, but a photo cannot determine its cause.',
  visibleFindings: ['small area of redness'],
  possibleCategories: ['dermatologic irritation'],
  photoFeedback: { usable: true, reason: 'Area is visible.', betterPhotoInstructions: [] },
  questions: ['Is it getting larger?'],
  ownerActions: ['Prevent repeated licking if this can be done comfortably.'],
  avoid: ['Do not apply human medication without veterinary guidance.'],
  vetHandoff: {
    recommended: false,
    timing: 'not-yet',
    summary: 'Monitor the localized change and seek veterinary care if it worsens or persists.',
    bring: [],
  },
};

const modelProvenance = {
  provider: 'openai' as const,
  model: 'health-model-snapshot',
  policyVersion: 'woof-health-model-policy-v2',
  responseContract: 'woof-pet-health-lens-json-schema-v1' as const,
};

type PrismaMock = {
  pet: { findFirst: jest.Mock };
  activity: { findMany: jest.Mock };
  telemetry: {
    findMany: jest.Mock;
    findFirst: jest.Mock;
    create: jest.Mock;
    delete: jest.Mock;
  };
};

type AiMock = {
  isConfigured: jest.Mock;
  analyze: jest.Mock;
  provenance: jest.Mock;
};

function createHarness() {
  const prisma: PrismaMock = {
    pet: { findFirst: jest.fn().mockResolvedValue(pet) },
    activity: { findMany: jest.fn().mockResolvedValue([]) },
    telemetry: {
      findMany: jest.fn().mockResolvedValue([]),
      findFirst: jest.fn(),
      create: jest.fn().mockResolvedValue({ id: 'assessment-1', createdAt: new Date() }),
      delete: jest.fn(),
    },
  };
  const ai: AiMock = {
    isConfigured: jest.fn().mockReturnValue(true),
    analyze: jest.fn().mockResolvedValue(safeModelResult),
    provenance: jest.fn().mockReturnValue(modelProvenance),
  };
  const service = new HealthLensService(
    prisma as unknown as PrismaService,
    ai as unknown as HealthAiService
  );
  return { service, prisma, ai };
}

describe('HealthLensService safety boundary', () => {
  it('short-circuits obvious emergencies before calling the model', async () => {
    const { service, ai } = createHarness();

    const result = await service.analyze('user-1', {
      petId: 'pet-1',
      concern: 'My dog is struggling to breathe and has blue gums',
      saveToTimeline: true,
    });

    expect(result.assessment.triage).toBe('emergency_now');
    expect(result.provenance.pathway).toBe('deterministic-emergency-screen');
    expect(result.provenance.model).toBeNull();
    expect(ai.analyze).not.toHaveBeenCalled();
    expect(ai.provenance).not.toHaveBeenCalled();
  });

  it('treats a structured major breathing change as an emergency without model arbitration', async () => {
    const { service, ai } = createHarness();

    const result = await service.analyze('user-1', {
      petId: 'pet-1',
      concern: 'Something is different tonight',
      breathing: 'major-change',
      saveToTimeline: false,
    });

    expect(result.assessment.triage).toBe('emergency_now');
    expect(result.provenance.model).toBeNull();
    expect(ai.analyze).not.toHaveBeenCalled();
  });

  it('fails closed instead of pretending to interpret a photo when the model is unavailable', async () => {
    const { service, ai } = createHarness();
    ai.isConfigured.mockReturnValue(false);

    const result = await service.analyze(
      'user-1',
      {
        petId: 'pet-1',
        concern: 'A small red patch appeared on the paw this morning',
        saveToTimeline: false,
      },
      createImage()
    );

    expect(result.assessment.triage).toBe('insufficient_information');
    expect(result.assessment.confidence).toBe(0);
    expect(result.provenance.imageAnalyzed).toBe(false);
    expect(result.provenance.model).toBeNull();
    expect(ai.analyze).not.toHaveBeenCalled();
  });

  it('records the exact model-policy identity for a model-backed assessment', async () => {
    const { service, prisma, ai } = createHarness();

    const result = await service.analyze('user-1', {
      petId: 'pet-1',
      concern: 'A small red patch appeared on the paw this morning',
      saveToTimeline: true,
    });

    expect(result.provenance.model).toEqual(modelProvenance);
    expect(ai.provenance).toHaveBeenCalledTimes(1);
    const call = prisma.telemetry.create.mock.calls[0][0] as {
      data: { data: Record<string, unknown> };
    };
    expect(call.data.data.model).toEqual(modelProvenance);
  });

  it('stores only derived assessment data and an irreversible image fingerprint', async () => {
    const { service, prisma } = createHarness();
    const image = createImage();

    const result = await service.analyze(
      'user-1',
      {
        petId: 'pet-1',
        concern: 'A small red patch appeared on the paw this morning',
        saveToTimeline: true,
      },
      image
    );

    expect(result.privacy.imageStoredByWoof).toBe(false);
    expect(prisma.telemetry.create).toHaveBeenCalledTimes(1);
    const call = prisma.telemetry.create.mock.calls[0][0] as {
      data: { data: Record<string, unknown> };
    };
    const stored = call.data.data;
    expect(stored.hadImage).toBe(true);
    expect(stored.imageSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(stored.model).toEqual(modelProvenance);
    expect(JSON.stringify(stored)).not.toContain(image.buffer.toString('base64'));
    expect(stored).not.toHaveProperty('image');
    expect(stored).not.toHaveProperty('imageUrl');
  });

  it('rejects timeline access when the pet is not owned by the actor', async () => {
    const { service, prisma } = createHarness();
    prisma.pet.findFirst.mockResolvedValue(null);

    await expect(service.timeline('user-1', 'pet-2')).rejects.toBeInstanceOf(ForbiddenException);
  });
});

function createImage(): Express.Multer.File {
  const buffer = Buffer.from('private-health-image-bytes');
  return {
    fieldname: 'image',
    originalname: 'paw.jpg',
    encoding: '7bit',
    mimetype: 'image/jpeg',
    size: buffer.length,
    buffer,
    stream: undefined as never,
    destination: '',
    filename: '',
    path: '',
  };
}
