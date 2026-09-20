import { ConflictException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TrustSafetyService } from '../trust-safety/trust-safety.service';
import { CaregiverOperationalStore } from './caregiver-operational.store';
import { CaregiverService } from './caregiver.service';
import { PetCapabilityAuthority } from './pet-capability-authority';

const now = new Date('2026-09-19T12:00:00.000Z');
const expiresAt = new Date('2026-09-19T13:00:00.000Z');

function dto(requestKey = 'caregiver-replay-key') {
  return {
    petId: 'pet-1',
    recipientUserId: 'caregiver-1',
    capabilities: ['VIEW_TODAY'] as const,
    expiresAt: expiresAt.toISOString(),
    requestKey,
  };
}

function grant(overrides: Record<string, unknown> = {}) {
  return {
    id: 'grant-1',
    petId: 'pet-1',
    issuerUserId: 'owner-1',
    recipientUserId: 'caregiver-1',
    requestKey: 'caregiver-replay-key',
    policyVersion: 'caregiver-authority-v1',
    status: 'PENDING_ACCEPTANCE' as const,
    issuedAt: now.toISOString(),
    acceptedAt: null,
    declinedAt: null,
    expiresAt: expiresAt.toISOString(),
    revokedAt: null,
    revokedByUserId: null,
    createdAt: now.toISOString(),
    updatedAt: now.toISOString(),
    capabilities: ['VIEW_TODAY'] as const,
    ...overrides,
  };
}

function harness() {
  const prisma = {
    user: {
      findUnique: jest.fn().mockResolvedValue({ id: 'caregiver-1' }),
    },
  };
  const store = {
    issueGrant: jest.fn(),
    getByIssuerRequestKey: jest.fn(),
    findLiveGrantForRecipientPet: jest.fn(),
    getIssuedGrant: jest.fn(),
  };
  const authority = {
    assertCanIssueGrant: jest.fn().mockResolvedValue(undefined),
  };
  const trustSafety = {
    isBlockedEitherDirection: jest.fn().mockResolvedValue(false),
  };

  return {
    store,
    service: new CaregiverService(
      prisma as unknown as PrismaService,
      store as unknown as CaregiverOperationalStore,
      authority as unknown as PetCapabilityAuthority,
      trustSafety as unknown as TrustSafetyService
    ),
  };
}

describe('CaregiverService issuance serialization', () => {
  it('resolves an exact concurrent retry only after the serialized store decision', async () => {
    const { service, store } = harness();
    store.issueGrant.mockResolvedValue(false);
    store.getByIssuerRequestKey.mockResolvedValue(grant());
    store.findLiveGrantForRecipientPet.mockResolvedValue(grant());

    const result = await service.issueGrant('owner-1', dto(), now);

    expect(result).toEqual(
      expect.objectContaining({
        id: 'grant-1',
        replayed: true,
        effectiveStatus: 'PENDING_ACCEPTANCE',
      })
    );
    expect(store.issueGrant).toHaveBeenCalledTimes(1);
    expect(store.getByIssuerRequestKey).toHaveBeenCalledTimes(1);
    expect(store.findLiveGrantForRecipientPet).not.toHaveBeenCalled();
    expect(store.issueGrant.mock.invocationCallOrder[0]).toBeLessThan(
      store.getByIssuerRequestKey.mock.invocationCallOrder[0]!
    );
  });

  it('maps a different live grant to conflict only after the serialized store no-op', async () => {
    const { service, store } = harness();
    store.issueGrant.mockResolvedValue(false);
    store.getByIssuerRequestKey.mockResolvedValue(null);
    store.findLiveGrantForRecipientPet.mockResolvedValue(
      grant({ id: 'other-grant', requestKey: 'other-request-key' })
    );

    await expect(service.issueGrant('owner-1', dto(), now)).rejects.toBeInstanceOf(
      ConflictException
    );

    expect(store.issueGrant).toHaveBeenCalledTimes(1);
    expect(store.getByIssuerRequestKey).toHaveBeenCalledTimes(1);
    expect(store.findLiveGrantForRecipientPet).toHaveBeenCalledTimes(1);
    expect(store.issueGrant.mock.invocationCallOrder[0]).toBeLessThan(
      store.getByIssuerRequestKey.mock.invocationCallOrder[0]!
    );
    expect(store.getByIssuerRequestKey.mock.invocationCallOrder[0]).toBeLessThan(
      store.findLiveGrantForRecipientPet.mock.invocationCallOrder[0]!
    );
  });
});
