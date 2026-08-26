import { BadRequestException, ConflictException } from '@nestjs/common';
import { createHash } from 'crypto';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { PetsService } from './pets.service';

describe('PetsService replay-safe creation', () => {
  const ownerId = '11111111-1111-4111-8111-111111111111';
  const householdId = '22222222-2222-4222-8222-222222222222';
  const creationKey = 'first-adventure:test-session';

  const expectedPetId = () => {
    const digest = createHash('sha256')
      .update(`woof-pet-create-v1:${ownerId}:${creationKey}`)
      .digest('hex');
    return `pet_${digest.slice(0, 32)}`;
  };

  const replayPet = (overrides: Record<string, unknown> = {}) => ({
    id: expectedPetId(),
    ownerId,
    name: 'Mochi',
    species: 'DOG',
    breed: 'Mix',
    sex: null,
    birthdate: new Date('2023-05-01'),
    owner: { id: ownerId, handle: 'sid', avatarUrl: null, isVerified: false },
    householdMemberships: [{ householdId }],
    ...overrides,
  });

  function createHarness() {
    const prisma = {
      pet: {
        findUnique: jest.fn(),
        create: jest.fn(),
        findMany: jest.fn(),
        count: jest.fn(),
        findFirst: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
      },
    };
    const households = {
      ensurePersonalHousehold: jest.fn().mockResolvedValue(householdId),
    };

    return {
      prisma,
      households,
      service: new PetsService(
        prisma as unknown as PrismaService,
        households as unknown as HouseholdsService
      ),
    };
  }

  it('converges an exact onboarding replay on one deterministic pet', async () => {
    const { prisma, service } = createHarness();
    const dto = {
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      birthdate: '2023-05-01',
      creationKey,
    };
    const created = replayPet();

    prisma.pet.findUnique.mockResolvedValueOnce(null).mockResolvedValueOnce(created);
    prisma.pet.create.mockResolvedValue(created);

    await expect(service.create(ownerId, dto)).resolves.toEqual(created);
    await expect(service.create(ownerId, dto)).resolves.toEqual(created);

    expect(prisma.pet.create).toHaveBeenCalledTimes(1);
    expect(prisma.pet.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ id: expectedPetId(), name: 'Mochi', species: 'DOG' }),
      })
    );
  });

  it('fails closed when the same creation key is replayed with different identity fields', async () => {
    const { prisma, service } = createHarness();
    prisma.pet.findUnique.mockResolvedValue(replayPet());

    await expect(
      service.create(ownerId, {
        name: 'Different dog',
        species: 'DOG',
        breed: 'Mix',
        birthdate: '2023-05-01',
        creationKey,
      })
    ).rejects.toBeInstanceOf(ConflictException);

    expect(prisma.pet.create).not.toHaveBeenCalled();
  });

  it('keeps media and mutable profile JSON out of replay-safe creation', async () => {
    const { prisma, service } = createHarness();

    await expect(
      service.create(ownerId, {
        name: 'Mochi',
        species: 'DOG',
        creationKey,
        avatarUrl: 'https://cdn.example.test/mochi.jpg',
      })
    ).rejects.toBeInstanceOf(BadRequestException);

    expect(prisma.pet.findUnique).not.toHaveBeenCalled();
    expect(prisma.pet.create).not.toHaveBeenCalled();
  });
});
