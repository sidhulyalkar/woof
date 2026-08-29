import {
  CAREGIVER_POLICY_VERSION,
  caregiverReceiptHash,
  effectiveCaregiverStatus,
  normalizeCaregiverCapabilities,
} from './caregiver.policy';

describe('caregiver authority policy', () => {
  it('derives expiry without mutating the stored lifecycle state', () => {
    const grant = {
      status: 'ACTIVE' as const,
      expiresAt: '2026-08-29T10:00:00.000Z',
    };

    expect(effectiveCaregiverStatus(grant, new Date('2026-08-29T09:59:59.999Z'))).toBe('ACTIVE');
    expect(effectiveCaregiverStatus(grant, new Date('2026-08-29T10:00:00.000Z'))).toBe('EXPIRED');
    expect(grant.status).toBe('ACTIVE');
  });

  it('does not reinterpret terminal stored state as expired', () => {
    expect(
      effectiveCaregiverStatus(
        { status: 'REVOKED', expiresAt: '2026-08-29T10:00:00.000Z' },
        new Date('2026-08-30T10:00:00.000Z')
      )
    ).toBe('REVOKED');
  });

  it('normalizes capability order and removes duplicates', () => {
    expect(normalizeCaregiverCapabilities(['LOG_OBSERVATION', 'VIEW_TODAY', 'VIEW_TODAY'])).toEqual(
      ['LOG_OBSERVATION', 'VIEW_TODAY']
    );
  });

  it('hashes the same receipt authority identically regardless of capability input order', () => {
    const base = {
      grantId: 'grant-1',
      petId: 'pet-1',
      issuerUserId: 'owner-1',
      recipientUserId: 'caregiver-1',
      transition: 'ISSUED' as const,
      actorUserId: 'owner-1',
      statusAfter: 'PENDING_ACCEPTANCE' as const,
      expiresAt: '2026-08-30T10:00:00.000Z',
      occurredAt: '2026-08-29T10:00:00.000Z',
      policyVersion: CAREGIVER_POLICY_VERSION,
    };

    expect(caregiverReceiptHash({ ...base, capabilities: ['VIEW_TODAY', 'LOG_OBSERVATION'] })).toBe(
      caregiverReceiptHash({ ...base, capabilities: ['LOG_OBSERVATION', 'VIEW_TODAY'] })
    );
  });

  it('changes the receipt hash when authority-relevant content changes', () => {
    const common = {
      grantId: 'grant-1',
      petId: 'pet-1',
      issuerUserId: 'owner-1',
      recipientUserId: 'caregiver-1',
      transition: 'ISSUED' as const,
      actorUserId: 'owner-1',
      statusAfter: 'PENDING_ACCEPTANCE' as const,
      capabilities: ['VIEW_TODAY'] as const,
      occurredAt: '2026-08-29T10:00:00.000Z',
      policyVersion: CAREGIVER_POLICY_VERSION,
    };

    expect(caregiverReceiptHash({ ...common, expiresAt: '2026-08-30T10:00:00.000Z' })).not.toBe(
      caregiverReceiptHash({ ...common, expiresAt: '2026-08-31T10:00:00.000Z' })
    );
  });
});
