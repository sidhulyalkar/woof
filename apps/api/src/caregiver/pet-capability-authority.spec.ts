import { NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TrustSafetyService } from '../trust-safety/trust-safety.service';
import { CaregiverOperationalStore } from './caregiver-operational.store';
import { PetCapabilityAuthority } from './pet-capability-authority';

function harness() {
  const prisma = {
    pet: {
      findFirst: jest.fn(),
    },
  };
  const store = {
    findEffectiveGrantForCapability: jest.fn(),
  };
  const trustSafety = {
    isBlockedEitherDirection: jest.fn().mockResolvedValue(false),
  };

  return {
    prisma,
    store,
    trustSafety,
    authority: new PetCapabilityAuthority(
      prisma as unknown as PrismaService,
      store as unknown as CaregiverOperationalStore,
      trustSafety as unknown as TrustSafetyService
    ),
  };
}

const grant = {
  id: 'grant-1',
  petId: 'pet-1',
  issuerUserId: 'owner-1',
  recipientUserId: 'caregiver-1',
  requestKey: 'request-1',
  policyVersion: 'caregiver-authority-v1',
  status: 'ACTIVE' as const,
  issuedAt: '2026-08-29T10:00:00.000Z',
  acceptedAt: '2026-08-29T10:01:00.000Z',
  declinedAt: null,
  expiresAt: '2026-08-30T10:00:00.000Z',
  revokedAt: null,
  revokedByUserId: null,
  createdAt: '2026-08-29T10:00:00.000Z',
  updatedAt: '2026-08-29T10:01:00.000Z',
  capabilities: ['VIEW_TODAY'] as const,
  pet: {
    id: 'pet-1',
    name: 'Nova',
    species: 'DOG',
    breed: 'Husky mix',
    avatarUrl: null,
  },
  issuerHandle: 'owner',
};

describe('PetCapabilityAuthority', () => {
  it('preserves existing owner/household authority without consulting caregiver grants', async () => {
    const { authority, prisma, store, trustSafety } = harness();
    prisma.pet.findFirst.mockResolvedValue({ id: 'pet-1' });

    await expect(authority.assertCapability('owner-1', 'pet-1', 'VIEW_TODAY')).resolves.toEqual({
      source: 'OWNER_OR_HOUSEHOLD',
      petId: 'pet-1',
      capability: 'VIEW_TODAY',
      caregiverGrant: null,
    });

    expect(store.findEffectiveGrantForCapability).not.toHaveBeenCalled();
    expect(trustSafety.isBlockedEitherDirection).not.toHaveBeenCalled();
  });

  it('resolves an exact active caregiver capability without converting it into household authority', async () => {
    const { authority, prisma, store, trustSafety } = harness();
    prisma.pet.findFirst.mockResolvedValue(null);
    store.findEffectiveGrantForCapability.mockResolvedValue(grant);

    await expect(authority.assertCapability('caregiver-1', 'pet-1', 'VIEW_TODAY')).resolves.toEqual(
      {
        source: 'CAREGIVER_GRANT',
        petId: 'pet-1',
        capability: 'VIEW_TODAY',
        caregiverGrant: grant,
      }
    );

    expect(trustSafety.isBlockedEitherDirection).toHaveBeenCalledWith('caregiver-1', 'owner-1');
  });

  it('fails closed when a relationship block exists even if a persisted grant is still active', async () => {
    const { authority, prisma, store, trustSafety } = harness();
    prisma.pet.findFirst.mockResolvedValue(null);
    store.findEffectiveGrantForCapability.mockResolvedValue(grant);
    trustSafety.isBlockedEitherDirection.mockResolvedValue(true);

    await expect(
      authority.assertCapability('caregiver-1', 'pet-1', 'VIEW_TODAY')
    ).rejects.toBeInstanceOf(NotFoundException);
  });

  it('does not let ordinary household-member semantics become grant-issuance authority', async () => {
    const { authority, prisma } = harness();
    prisma.pet.findFirst.mockResolvedValue(null);

    await expect(authority.assertCanIssueGrant('member-1', 'pet-1')).rejects.toBeInstanceOf(
      NotFoundException
    );

    expect(prisma.pet.findFirst).toHaveBeenCalledWith(
      expect.objectContaining({
        where: expect.objectContaining({
          id: 'pet-1',
          OR: expect.arrayContaining([
            expect.objectContaining({ ownerId: 'member-1' }),
            expect.objectContaining({
              householdMemberships: expect.objectContaining({
                some: expect.objectContaining({
                  household: expect.objectContaining({
                    members: expect.objectContaining({
                      some: expect.objectContaining({
                        userId: 'member-1',
                        role: { in: ['OWNER', 'ADMIN'] },
                      }),
                    }),
                  }),
                }),
              }),
            }),
          ]),
        }),
      })
    );
  });
});
