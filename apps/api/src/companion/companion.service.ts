import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import type { UpdateReadinessReflectionDto } from './dto/companion.dto';
import {
  READINESS_DIMENSIONS,
  resolveCompanionState,
  type CompanionMode,
  type ReadinessDimension,
  type ReadinessStatus,
} from './companion.policy';

type ModeRow = {
  mode: string | null;
};

type ReadinessRow = {
  dimensions: unknown;
  updatedAt: Date;
};

@Injectable()
export class CompanionService {
  constructor(private readonly prisma: PrismaService) {}

  async getState(userId: string) {
    const [mode, hasAuthorizedPet] = await Promise.all([
      this.getPersistedMode(userId),
      this.hasAuthorizedPet(userId),
    ]);
    const state = resolveCompanionState(mode, hasAuthorizedPet);

    return {
      ...state,
      authority: {
        modeControlsPresentation: true,
        petAccessComesFromRelationships: true,
        modeNeverCreatesPetAuthority: true,
      },
    };
  }

  async updateMode(userId: string, mode: CompanionMode) {
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_companion.profiles (user_id, mode, updated_at)
      VALUES (${userId}, ${mode}, NOW())
      ON CONFLICT (user_id)
      DO UPDATE SET mode = EXCLUDED.mode, updated_at = NOW()
    `);
    return this.getState(userId);
  }

  async getReadiness(userId: string) {
    const rows = await this.prisma.$queryRaw<ReadinessRow[]>(Prisma.sql`
      SELECT dimensions, updated_at AS "updatedAt"
      FROM dogos_companion.readiness_reflections
      WHERE user_id = ${userId}
      LIMIT 1
    `);
    const row = rows[0];
    const dimensions = this.readinessDimensions(row?.dimensions);

    return {
      dimensions,
      updatedAt: row?.updatedAt.toISOString() ?? null,
      statuses: ['NOT_SURE', 'WORKING_ON_IT', 'READY_TO_DISCUSS'] as const,
      disclaimer:
        'This is a private reflection, not an adoption, foster, financial, housing, or welfare assessment. Woof does not combine these answers into a readiness score.',
    };
  }

  async updateReadiness(userId: string, dto: UpdateReadinessReflectionDto) {
    const patch = Object.fromEntries(
      Object.entries(dto).filter(([, value]) => value !== undefined)
    ) as Partial<Record<ReadinessDimension, ReadinessStatus>>;

    if (Object.keys(patch).length > 0) {
      await this.prisma.$executeRaw(Prisma.sql`
        INSERT INTO dogos_companion.readiness_reflections (user_id, dimensions, updated_at)
        VALUES (${userId}, ${JSON.stringify(patch)}::jsonb, NOW())
        ON CONFLICT (user_id)
        DO UPDATE SET
          dimensions = dogos_companion.readiness_reflections.dimensions || EXCLUDED.dimensions,
          updated_at = NOW()
      `);
    }

    return this.getReadiness(userId);
  }

  private async getPersistedMode(userId: string): Promise<CompanionMode | null> {
    const rows = await this.prisma.$queryRaw<ModeRow[]>(Prisma.sql`
      SELECT mode
      FROM dogos_companion.profiles
      WHERE user_id = ${userId}
      LIMIT 1
    `);
    const mode = rows[0]?.mode ?? null;
    return mode === 'PET_GUARDIAN' || mode === 'ANIMAL_ALLY' || mode === 'FOSTER_CAREGIVER'
      ? mode
      : null;
  }

  private async hasAuthorizedPet(userId: string): Promise<boolean> {
    const count = await this.prisma.pet.count({
      where: {
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
                    },
                  },
                },
              },
            },
          },
        ],
      },
    });
    return count > 0;
  }

  private readinessDimensions(value: unknown) {
    const record =
      value && typeof value === 'object' && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : {};

    return Object.fromEntries(
      READINESS_DIMENSIONS.map((dimension) => {
        const status = record[dimension];
        const valid =
          status === 'NOT_SURE' || status === 'WORKING_ON_IT' || status === 'READY_TO_DISCUSS';
        return [dimension, valid ? status : null];
      })
    ) as Record<ReadinessDimension, ReadinessStatus | null>;
  }
}
