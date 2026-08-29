import {
  BadRequestException,
  ConflictException,
  ForbiddenException,
  Injectable,
  NotFoundException,
} from '@nestjs/common';
import { randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import { TrustSafetyService } from '../trust-safety/trust-safety.service';
import { CaregiverOperationalStore } from './caregiver-operational.store';
import {
  CAREGIVER_MAX_GRANT_MS,
  CAREGIVER_MIN_GRANT_MS,
  effectiveCaregiverStatus,
  normalizeCaregiverCapabilities,
  type CaregiverCapability,
} from './caregiver.policy';
import type { CreateCaregiverObservationDto, IssueCaregiverGrantDto } from './dto/caregiver.dto';
import { PetCapabilityAuthority } from './pet-capability-authority';

type GrantView = NonNullable<
  Awaited<ReturnType<CaregiverOperationalStore['getReceivedGrant']>>
>;

@Injectable()
export class CaregiverService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly store: CaregiverOperationalStore,
    private readonly authority: PetCapabilityAuthority,
    private readonly trustSafety: TrustSafetyService,
  ) {}

  async issueGrant(issuerUserId: string, dto: IssueCaregiverGrantDto, now = new Date()) {
    const requestKey = dto.requestKey.trim();
    if (requestKey.length < 8) throw new BadRequestException('Request key is too short');

    const capabilities = normalizeCaregiverCapabilities(dto.capabilities);
    this.assertCapabilityBundle(capabilities);

    const expiresAt = new Date(dto.expiresAt);
    const durationMs = expiresAt.getTime() - now.getTime();
    if (!Number.isFinite(expiresAt.getTime())) {
      throw new BadRequestException('Grant expiry must be a valid instant');
    }
    if (durationMs < CAREGIVER_MIN_GRANT_MS || durationMs > CAREGIVER_MAX_GRANT_MS) {
      throw new BadRequestException('Caregiver access must last between 15 minutes and 31 days');
    }
    if (issuerUserId === dto.recipientUserId) {
      throw new BadRequestException('Caregiver access cannot be granted to yourself');
    }

    await this.authority.assertCanIssueGrant(issuerUserId, dto.petId);

    const recipient = await this.prisma.user.findUnique({
      where: { id: dto.recipientUserId },
      select: { id: true },
    });
    if (!recipient) throw new NotFoundException('Member not found');
    if (await this.trustSafety.isBlockedEitherDirection(issuerUserId, dto.recipientUserId)) {
      throw new ForbiddenException('Caregiver access cannot be created for this relationship');
    }

    const replay = await this.store.getByIssuerRequestKey(issuerUserId, requestKey);
    if (replay) {
      if (this.matchesIssuance(replay, dto, capabilities, expiresAt)) {
        return { ...this.grantView(replay, now), replayed: true };
      }
      throw new ConflictException('Request key was already used for a different caregiver grant');
    }

    const existing = await this.store.findLiveGrantForRecipientPet(dto.recipientUserId, dto.petId);
    if (existing && effectiveCaregiverStatus(existing, now) !== 'EXPIRED') {
      throw new ConflictException('This caregiver already has pending or active access to this pet');
    }

    const grantId = randomUUID();
    try {
      const created = await this.store.issueGrant({
        id: grantId,
        petId: dto.petId,
        issuerUserId,
        recipientUserId: dto.recipientUserId,
        requestKey,
        capabilities,
        issuedAt: now,
        expiresAt,
      });
      if (!created) return this.resolveIssuanceNoop(issuerUserId, dto, requestKey, capabilities, expiresAt, now);
    } catch (error) {
      const semantic = await this.resolveIssuanceNoop(
        issuerUserId,
        dto,
        requestKey,
        capabilities,
        expiresAt,
        now,
        false,
      );
      if (semantic) return semantic;
      throw error;
    }

    const grant = await this.store.getIssuedGrant(issuerUserId, grantId);
    if (!grant) throw new ConflictException('Caregiver grant could not be read after issuance');
    return { ...this.grantView(grant, now), replayed: false };
  }

  async listIssued(userId: string, now = new Date()) {
    const grants = await this.store.listIssuedGrants(userId);
    return grants.map((grant) => this.grantView(grant, now));
  }

  async listReceived(userId: string, now = new Date()) {
    const grants = await this.store.listReceivedGrants(userId);
    return Promise.all(
      grants.map(async (grant) => ({
        ...this.grantView(grant, now),
        relationshipBlocked: await this.trustSafety.isBlockedEitherDirection(
          grant.issuerUserId,
          grant.recipientUserId,
        ),
      })),
    );
  }

  async acceptGrant(recipientUserId: string, grantId: string, now = new Date()) {
    const before = await this.requireReceivedGrant(recipientUserId, grantId);
    const effective = effectiveCaregiverStatus(before, now);
    if (effective === 'ACTIVE') return { ...this.grantView(before, now), replayed: true };
    if (effective === 'EXPIRED') throw new ConflictException('Caregiver invitation has expired');
    if (before.status !== 'PENDING_ACCEPTANCE') {
      throw new ConflictException('Caregiver invitation is no longer pending');
    }
    if (await this.trustSafety.isBlockedEitherDirection(before.issuerUserId, recipientUserId)) {
      throw new ForbiddenException('Caregiver access cannot be accepted for this relationship');
    }

    const changed = await this.store.acceptGrant(grantId, recipientUserId, now);
    const after = await this.requireReceivedGrant(recipientUserId, grantId);
    if (!changed) {
      if (effectiveCaregiverStatus(after, now) === 'ACTIVE') {
        return { ...this.grantView(after, now), replayed: true };
      }
      throw new ConflictException('Caregiver invitation changed before it could be accepted');
    }
    return { ...this.grantView(after, now), replayed: false };
  }

  async declineGrant(recipientUserId: string, grantId: string, now = new Date()) {
    const before = await this.requireReceivedGrant(recipientUserId, grantId);
    const effective = effectiveCaregiverStatus(before, now);
    if (before.status === 'DECLINED') return { ...this.grantView(before, now), replayed: true };
    if (effective === 'EXPIRED') throw new ConflictException('Caregiver invitation has expired');
    if (before.status !== 'PENDING_ACCEPTANCE') {
      throw new ConflictException('Caregiver invitation is no longer pending');
    }

    const changed = await this.store.declineGrant(grantId, recipientUserId, now);
    const after = await this.requireReceivedGrant(recipientUserId, grantId);
    if (!changed) {
      if (after.status === 'DECLINED') return { ...this.grantView(after, now), replayed: true };
      throw new ConflictException('Caregiver invitation changed before it could be declined');
    }
    return { ...this.grantView(after, now), replayed: false };
  }

  async revokeGrant(issuerUserId: string, grantId: string, now = new Date()) {
    const before = await this.requireIssuedGrant(issuerUserId, grantId);
    if (before.status === 'REVOKED') return { ...this.grantView(before, now), replayed: true };
    if (effectiveCaregiverStatus(before, now) === 'EXPIRED') {
      throw new ConflictException('Caregiver access has already expired');
    }
    if (!['PENDING_ACCEPTANCE', 'ACTIVE'].includes(before.status)) {
      throw new ConflictException('Caregiver access can no longer be revoked');
    }

    const changed = await this.store.revokeGrant(grantId, issuerUserId, now);
    const after = await this.requireIssuedGrant(issuerUserId, grantId);
    if (!changed) {
      if (after.status === 'REVOKED') return { ...this.grantView(after, now), replayed: true };
      throw new ConflictException('Caregiver access changed before it could be revoked');
    }
    return { ...this.grantView(after, now), replayed: false };
  }

  async listCaregiverPets(recipientUserId: string, now = new Date()) {
    const grants = await this.store.listActiveCaregiverPets(recipientUserId, now);
    const visible = [];
    for (const grant of grants) {
      if (await this.trustSafety.isBlockedEitherDirection(grant.issuerUserId, recipientUserId)) continue;
      visible.push(this.grantView(grant, now));
    }
    return visible;
  }

  async getCaregiverToday(recipientUserId: string, petId: string, now = new Date()) {
    const grant = await this.authority.assertCaregiverGrantCapability(
      recipientUserId,
      petId,
      'VIEW_TODAY',
      now,
    );

    return {
      pet: grant.pet,
      relationship: {
        grantId: grant.id,
        issuerUserId: grant.issuerUserId,
        issuerHandle: grant.issuerHandle,
        expiresAt: grant.expiresAt,
        capabilities: grant.capabilities,
        effectiveStatus: effectiveCaregiverStatus(grant, now),
      },
      available: {
        viewToday: true,
        logObservation: grant.capabilities.includes('LOG_OBSERVATION'),
      },
      boundaries: {
        householdHistory: false,
        siblingPets: false,
        medicalAuthority: false,
        profileCorrection: false,
        connectorAdmin: false,
        bondXpAuthority: false,
        recommendationEvidenceAuthority: false,
      },
    };
  }

  async logObservation(
    recipientUserId: string,
    petId: string,
    dto: CreateCaregiverObservationDto,
    now = new Date(),
  ) {
    const grant = await this.authority.assertCaregiverGrantCapability(
      recipientUserId,
      petId,
      'LOG_OBSERVATION',
      now,
    );
    const observedAt = dto.observedAt ? new Date(dto.observedAt) : now;
    if (!Number.isFinite(observedAt.getTime())) {
      throw new BadRequestException('Observation time must be a valid instant');
    }

    const summary = dto.summary.trim();
    if (!summary) throw new BadRequestException('Observation summary is required');
    const note = dto.note?.trim() || null;

    return this.store.recordObservation({
      grantId: grant.id,
      petId,
      actorUserId: recipientUserId,
      kind: dto.kind,
      summary,
      note,
      observedAt,
      context: {
        authorityClass: 'CONTEXT_ONLY',
        policyVersion: grant.policyVersion,
      },
    });
  }

  private grantView(grant: GrantView, now: Date) {
    return {
      ...grant,
      effectiveStatus: effectiveCaregiverStatus(grant, now),
    };
  }

  private async requireReceivedGrant(userId: string, grantId: string) {
    const grant = await this.store.getReceivedGrant(userId, grantId);
    if (!grant) throw new NotFoundException('Caregiver grant not found');
    return grant;
  }

  private async requireIssuedGrant(userId: string, grantId: string) {
    const grant = await this.store.getIssuedGrant(userId, grantId);
    if (!grant) throw new NotFoundException('Caregiver grant not found');
    return grant;
  }

  private assertCapabilityBundle(capabilities: CaregiverCapability[]) {
    if (capabilities.length === 0) throw new BadRequestException('At least one capability is required');
    if (capabilities.includes('LOG_OBSERVATION') && !capabilities.includes('VIEW_TODAY')) {
      throw new BadRequestException('LOG_OBSERVATION requires VIEW_TODAY');
    }
  }

  private matchesIssuance(
    existing: GrantView,
    dto: IssueCaregiverGrantDto,
    capabilities: CaregiverCapability[],
    expiresAt: Date,
  ) {
    return (
      existing.petId === dto.petId &&
      existing.recipientUserId === dto.recipientUserId &&
      existing.expiresAt === expiresAt.toISOString() &&
      existing.capabilities.length === capabilities.length &&
      existing.capabilities.every((capability, index) => capability === capabilities[index])
    );
  }

  private async resolveIssuanceNoop(
    issuerUserId: string,
    dto: IssueCaregiverGrantDto,
    requestKey: string,
    capabilities: CaregiverCapability[],
    expiresAt: Date,
    now: Date,
    throwOnConflict = true,
  ) {
    const replay = await this.store.getByIssuerRequestKey(issuerUserId, requestKey);
    if (replay) {
      if (this.matchesIssuance(replay, dto, capabilities, expiresAt)) {
        return { ...this.grantView(replay, now), replayed: true };
      }
      if (throwOnConflict) {
        throw new ConflictException('Request key was already used for a different caregiver grant');
      }
      return null;
    }

    const live = await this.store.findLiveGrantForRecipientPet(dto.recipientUserId, dto.petId);
    if (live && effectiveCaregiverStatus(live, now) !== 'EXPIRED') {
      if (throwOnConflict) {
        throw new ConflictException('This caregiver already has pending or active access to this pet');
      }
      return null;
    }

    if (throwOnConflict) throw new ConflictException('Caregiver grant was not created');
    return null;
  }
}
