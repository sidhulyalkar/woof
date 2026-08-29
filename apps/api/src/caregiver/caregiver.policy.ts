import { createHash } from 'node:crypto';

export const CAREGIVER_POLICY_VERSION = 'caregiver-authority-v1' as const;

export const CAREGIVER_CAPABILITIES = ['VIEW_TODAY', 'LOG_OBSERVATION'] as const;
export type CaregiverCapability = (typeof CAREGIVER_CAPABILITIES)[number];

export const CAREGIVER_STORED_STATUSES = [
  'PENDING_ACCEPTANCE',
  'ACTIVE',
  'DECLINED',
  'REVOKED',
] as const;
export type CaregiverStoredStatus = (typeof CAREGIVER_STORED_STATUSES)[number];
export type CaregiverEffectiveStatus = CaregiverStoredStatus | 'EXPIRED';

export const CAREGIVER_OBSERVATION_KINDS = [
  'ROUTINE',
  'ACTIVITY_RESPONSE',
  'BEHAVIOR',
  'HANDOFF_NOTE',
] as const;
export type CaregiverObservationKind = (typeof CAREGIVER_OBSERVATION_KINDS)[number];

export const CAREGIVER_AUTHORITY_CLASS = 'CONTEXT_ONLY' as const;
export const CAREGIVER_MAX_GRANT_MS = 31 * 24 * 60 * 60 * 1000;
export const CAREGIVER_MIN_GRANT_MS = 15 * 60 * 1000;
export const CAREGIVER_MAX_FUTURE_OBSERVATION_MS = 5 * 60 * 1000;

export type CaregiverGrantSnapshot = {
  id: string;
  petId: string;
  issuerUserId: string;
  recipientUserId: string;
  requestKey: string;
  policyVersion: string;
  status: CaregiverStoredStatus;
  issuedAt: string;
  acceptedAt: string | null;
  declinedAt: string | null;
  expiresAt: string;
  revokedAt: string | null;
  revokedByUserId: string | null;
  capabilities: CaregiverCapability[];
};

export function effectiveCaregiverStatus(
  grant: Pick<CaregiverGrantSnapshot, 'status' | 'expiresAt'>,
  now = new Date(),
): CaregiverEffectiveStatus {
  if (
    (grant.status === 'PENDING_ACCEPTANCE' || grant.status === 'ACTIVE') &&
    new Date(grant.expiresAt).getTime() <= now.getTime()
  ) {
    return 'EXPIRED';
  }
  return grant.status;
}

export function normalizeCaregiverCapabilities(values: readonly CaregiverCapability[]) {
  return [...new Set(values)].sort() as CaregiverCapability[];
}

export function caregiverReceiptHash(input: {
  grantId: string;
  petId: string;
  issuerUserId: string;
  recipientUserId: string;
  transition: 'ISSUED' | 'ACCEPTED' | 'DECLINED' | 'REVOKED';
  actorUserId: string;
  statusAfter: CaregiverStoredStatus;
  capabilities: readonly CaregiverCapability[];
  expiresAt: string;
  occurredAt: string;
  policyVersion: string;
}) {
  return createHash('sha256')
    .update(
      JSON.stringify({
        grantId: input.grantId,
        petId: input.petId,
        issuerUserId: input.issuerUserId,
        recipientUserId: input.recipientUserId,
        transition: input.transition,
        actorUserId: input.actorUserId,
        statusAfter: input.statusAfter,
        capabilities: normalizeCaregiverCapabilities(input.capabilities),
        expiresAt: input.expiresAt,
        occurredAt: input.occurredAt,
        policyVersion: input.policyVersion,
      }),
    )
    .digest('hex');
}
