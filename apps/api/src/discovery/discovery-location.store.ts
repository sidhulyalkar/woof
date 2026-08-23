import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type LocationRow = {
  user_id: string;
  lat_bucket: number;
  lng_bucket: number;
  precision_m: number;
  enabled: boolean;
  expires_at: Date;
  updated_at: Date;
};

@Injectable()
export class DiscoveryLocationStore {
  constructor(private readonly prisma: PrismaService) {}

  async upsert(input: {
    userId: string;
    latBucket: number;
    lngBucket: number;
    precisionM: number;
    expiresAt: Date;
  }) {
    const rows = await this.prisma.$queryRaw<LocationRow[]>(Prisma.sql`
      INSERT INTO dogos_discovery.locations (
        user_id, lat_bucket, lng_bucket, precision_m, enabled, expires_at, updated_at
      ) VALUES (
        CAST(${input.userId} AS uuid), ${input.latBucket}, ${input.lngBucket},
        ${input.precisionM}, TRUE, ${input.expiresAt}, NOW()
      )
      ON CONFLICT (user_id) DO UPDATE SET
        lat_bucket = EXCLUDED.lat_bucket,
        lng_bucket = EXCLUDED.lng_bucket,
        precision_m = EXCLUDED.precision_m,
        enabled = TRUE,
        expires_at = EXCLUDED.expires_at,
        updated_at = NOW()
      RETURNING user_id, lat_bucket, lng_bucket, precision_m, enabled, expires_at, updated_at
    `);
    return rows[0]!;
  }

  async disable(userId: string) {
    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_discovery.locations
      SET enabled = FALSE, updated_at = NOW()
      WHERE user_id = CAST(${userId} AS uuid)
    `);
  }

  async get(userId: string) {
    const rows = await this.prisma.$queryRaw<LocationRow[]>(Prisma.sql`
      SELECT user_id, lat_bucket, lng_bucket, precision_m, enabled, expires_at, updated_at
      FROM dogos_discovery.locations
      WHERE user_id = CAST(${userId} AS uuid)
      LIMIT 1
    `);
    return rows[0] ?? null;
  }

  async findNearby(input: {
    userId: string;
    minLatBucket: number;
    maxLatBucket: number;
    minLngBucket: number;
    maxLngBucket: number;
    take: number;
  }) {
    return this.prisma.$queryRaw<LocationRow[]>(Prisma.sql`
      SELECT user_id, lat_bucket, lng_bucket, precision_m, enabled, expires_at, updated_at
      FROM dogos_discovery.locations
      WHERE user_id <> CAST(${input.userId} AS uuid)
        AND enabled = TRUE
        AND expires_at > NOW()
        AND lat_bucket BETWEEN ${input.minLatBucket} AND ${input.maxLatBucket}
        AND lng_bucket BETWEEN ${input.minLngBucket} AND ${input.maxLngBucket}
      ORDER BY updated_at DESC
      LIMIT ${input.take}
    `);
  }
}
