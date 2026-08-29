import { Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { TrustSafetyService } from '../trust-safety/trust-safety.service';
import { CaregiverOperationalStore } from './caregiver-operational.store';
import type { CaregiverCapability } from './caregiver.policy';

export type PetCapabilityDecision =
  | {
      source: 'OWNER_OR_HOUSEHOLD';
      petId: string;
      capability: CaregiverCapability;
      caregiverGrant: null;
    }
  | {
      source: 'CAREGIVER_GRANT';
      petId: string;
      capability: CaregiverCapability;
      caregiverGrant: NonNullable<
        Awaited<ReturnType<CaregiverOperationalStore['findEffectiveGrantForCapability']>>
      >;
    };

@Injectable()
export class PetCapabilityAuthority {
  constructor(
    private readonly prisma: PrismaService,
    private readonly caregiverStore: CaregiverOperationalStore,
    private readonly trustSafety: TrustSafetyService
  ) {}

  /**
   * Resolve present-tense pet capability. Existing owner/active-household
   * semantics remain intact; explicit caregiver grants layer beside them.
   * This intentionally does not grant household history, admin, medical,
   * connector, reward, or pet-profile correction authority.
   */
  async assertCapability(
    userId: string,
    petId: string,
    capability: CaregiverCapability,
    now = new Date()
  ): Promise<PetCapabilityDecision> {
    const householdPet = await this.prisma.pet.findFirst({
      where: {
        id: petId,
        OR: [
          { ownerId: userId },
          {
            householdMemberships: {
              some: {
                status: 'ACTIVE',
                household: {
                  members: {
                    some: { userId, status: 'ACTIVE' },
                  },
                },
              },
            },
          },
        ],
      },
      select: { id: true },
    });

    if (householdPet) {
      return {
        source: 'OWNER_OR_HOUSEHOLD',
        petId,
        capability,
        caregiverGrant: null,
      };
    }

    const caregiverGrant = await this.caregiverStore.findEffectiveGrantForCapability({
      recipientUserId: userId,
      petId,
      capability,
      now,
    });
    if (!caregiverGrant) throw new NotFoundException('Pet not found');
    if (await this.trustSafety.isBlockedEitherDirection(userId, caregiverGrant.issuerUserId)) {
      throw new NotFoundException('Pet not found');
    }

    return {
      source: 'CAREGIVER_GRANT',
      petId,
      capability,
      caregiverGrant,
    };
  }

  /**
   * Issuance is deliberately stronger than ordinary household pet access.
   * Only the pet's owner or an active OWNER/ADMIN of an active household that
   * contains the pet may delegate temporary caregiver authority.
   */
  async assertCanIssueGrant(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: {
        id: petId,
        OR: [
          { ownerId: userId },
          {
            householdMemberships: {
              some: {
                status: 'ACTIVE',
                household: {
                  members: {
                    some: {
                      userId,
                      status: 'ACTIVE',
                      role: { in: ['OWNER', 'ADMIN'] },
                    },
                  },
                },
              },
            },
          },
        ],
      },
      select: { id: true, ownerId: true },
    });

    if (!pet) throw new NotFoundException('Pet not found');
    return pet;
  }

  /** Caregiver-specific surfaces must prove the explicit grant source. */
  async assertCaregiverGrantCapability(
    userId: string,
    petId: string,
    capability: CaregiverCapability,
    now = new Date()
  ) {
    const grant = await this.caregiverStore.findEffectiveGrantForCapability({
      recipientUserId: userId,
      petId,
      capability,
      now,
    });
    if (!grant) throw new NotFoundException('Caregiver pet access not found');
    if (await this.trustSafety.isBlockedEitherDirection(userId, grant.issuerUserId)) {
      throw new NotFoundException('Caregiver pet access not found');
    }
    return grant;
  }
}
