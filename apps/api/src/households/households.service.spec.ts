import { BadRequestException, ForbiddenException } from '@nestjs/common';
import { createHash } from 'crypto';
import { PrismaService } from '../prisma/prisma.service';
import { HouseholdsService } from './households.service';

describe('HouseholdsService', () => {
  const userId = '11111111-1111-4111-8111-111111111111';
  const petA = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  const petB = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb';

  const deterministicHouseholdId = (id: string) => {
    const hex = createHash('md5').update(`dogos-household:${id}`).digest('hex');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  };

  function createHarness() {
    const tx = {
      $queryRaw: jest.fn().mockResolvedValue([{ acquired: 1 }]),
      household: {
        upsert: jest.fn().mockResolvedValue({}),
      },
      householdMember: {
        upsert: jest.fn().mockResolvedValue({}),
      },
      householdPet: {
        upsert: jest.fn().mockResolvedValue({}),
      },
      pet: {
        findMany: jest.fn().mockResolvedValue([]),
      },
    };

    const prisma = {
      householdMember: {
        findUnique: jest.fn(),
        findMany: jest.fn(),
        findFirst: jest.fn(),
      },
      householdPet: {
        findUnique: jest.fn(),
        upsert: jest.fn(),
        update: jest.fn(),
      },
      household: {
        update: jest.fn(),
      },
      pet: {
        findFirst: jest.fn(),
      },
      $transaction: jest.fn(async (callback: (client: typeof tx) => Promise<unknown>) =>
        callback(tx)
      ),
    };

    return {
      tx,
      prisma,
      service: new HouseholdsService(prisma as unknown as PrismaService),
    };
  }

  it('always resolves the deterministic personal household, not an arbitrary membership', async () => {
    const { prisma, service } = createHarness();
    const expectedId = deterministicHouseholdId(userId);
    prisma.householdMember.findUnique.mockResolvedValue({ status: 'ACTIVE', role: 'OWNER' });

    await expect(service.ensurePersonalHousehold(userId)).resolves.toBe(expectedId);
    expect(prisma.householdMember.findUnique).toHaveBeenCalledWith({
      where: {
        householdId_userId: {
          householdId: expectedId,
          userId,
        },
      },
      select: { status: true, role: true },
    });
    expect(prisma.$transaction).not.toHaveBeenCalled();
  });

  it('serializes bootstrap before creating the owner and existing pets', async () => {
    const { prisma, service, tx } = createHarness();
    const expectedId = deterministicHouseholdId(userId);
    prisma.householdMember.findUnique.mockResolvedValue(null);
    tx.pet.findMany.mockResolvedValue([{ id: petA }, { id: petB }]);

    await expect(service.ensurePersonalHousehold(userId)).resolves.toBe(expectedId);

    expect(tx.$queryRaw).toHaveBeenCalledTimes(1);
    expect(tx.$queryRaw.mock.invocationCallOrder[0]).toBeLessThan(
      tx.household.upsert.mock.invocationCallOrder[0]!
    );
    expect(tx.household.upsert).toHaveBeenCalledWith({
      where: { id: expectedId },
      update: {},
      create: { id: expectedId, name: 'My household' },
    });
    expect(tx.householdMember.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        where: {
          householdId_userId: {
            householdId: expectedId,
            userId,
          },
        },
      })
    );
    expect(tx.householdPet.upsert).toHaveBeenCalledTimes(2);
  });

  it('selects one household only when every chosen pet participates in that household', async () => {
    const { prisma, service } = createHarness();
    prisma.householdMember.findMany.mockResolvedValue([
      {
        householdId: 'household-together',
        household: { pets: [{ petId: petA }, { petId: petB }] },
      },
    ]);

    await expect(service.resolveActivityHousehold(userId, [petA, petB])).resolves.toBe(
      'household-together'
    );
  });

  it('rejects a fake shared activity when the selected pets only exist in separate households', async () => {
    const { prisma, service } = createHarness();
    prisma.householdMember.findMany.mockResolvedValue([
      { householdId: 'household-a', household: { pets: [{ petId: petA }] } },
      { householdId: 'household-b', household: { pets: [{ petId: petB }] } },
    ]);
    prisma.pet.findFirst.mockResolvedValue({ id: petA });

    await expect(service.resolveActivityHousehold(userId, [petA, petB])).rejects.toBeInstanceOf(
      BadRequestException
    );
  });

  it('keeps household mutations manager-only', async () => {
    const { prisma, service } = createHarness();
    prisma.householdMember.findFirst.mockResolvedValue({ role: 'MEMBER' });

    await expect(service.update(userId, 'household-a', { name: 'Pack HQ' })).rejects.toBeInstanceOf(
      ForbiddenException
    );
    expect(prisma.household.update).not.toHaveBeenCalled();
  });
});
