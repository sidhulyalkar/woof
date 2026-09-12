import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import {
  PACK_COARSE_REGIONS,
  PACK_LOCALITY_CONTRACT,
  type PackCoarseRegionId,
} from './pack-locality.catalog';

type PackLocalityRow = {
  id: string;
  ownerUserId: string;
  scope: string;
  regionKey: string | null;
};

type PackLocalityShape = {
  id: string;
  regionKey: string | null;
};

type PackCatalogRow = PackLocalityShape & {
  name: string;
  slug: string;
  scope: string;
  visibility: string;
  memberCount: number;
  joined: boolean;
  role: string | null;
};

export type PackCatalogLike = {
  packs: PackCatalogRow[];
  localMinimumCohort: number;
  locationContract: string;
};

@Injectable()
export class PackLocalityService {
  constructor(private readonly prisma: PrismaService) {}

  getRegions() {
    return {
      regions: PACK_COARSE_REGIONS,
      locationContract: PACK_LOCALITY_CONTRACT,
    };
  }

  decorateCatalog(catalog: PackCatalogLike) {
    return {
      ...catalog,
      packs: catalog.packs
        .filter((pack) => pack.joined || this.isApprovedRegionId(pack.regionKey))
        .map((pack) => this.decoratePack(pack)),
      locationContract: PACK_LOCALITY_CONTRACT,
    };
  }

  decoratePack<T extends PackLocalityShape>(pack: T) {
    const coarseRegion = this.regionById(pack.regionKey);
    return {
      ...pack,
      localityStatus: coarseRegion ? ('APPROVED' as const) : ('LEGACY_UNVERIFIED' as const),
      coarseRegion,
    };
  }

  async requireLocalityAuthority(packId: string): Promise<void> {
    const rows = await this.prisma.$queryRaw<PackLocalityRow[]>(Prisma.sql`
      SELECT
        id,
        owner_user_id AS "ownerUserId",
        scope,
        region_key AS "regionKey"
      FROM dogos_social.packs
      WHERE id = ${packId}
      LIMIT 1
    `);
    const pack = rows[0];
    if (!pack) throw new NotFoundException('Pack not found');
    if (pack.scope === 'LOCAL' && !this.isApprovedRegionId(pack.regionKey)) {
      throw new BadRequestException(
        'This legacy Pack needs an approved broad locality before discovery, joining, or local standings are available.'
      );
    }
  }

  async repairLocality(userId: string, packId: string, regionKey: PackCoarseRegionId) {
    if (!this.isApprovedRegionId(regionKey)) {
      throw new BadRequestException('Choose an approved broad locality');
    }

    const rows = await this.prisma.$queryRaw<PackLocalityRow[]>(Prisma.sql`
      SELECT
        id,
        owner_user_id AS "ownerUserId",
        scope,
        region_key AS "regionKey"
      FROM dogos_social.packs
      WHERE id = ${packId}
      LIMIT 1
    `);
    const pack = rows[0];
    if (!pack || pack.ownerUserId !== userId) {
      // Keep Pack ownership non-enumerable through this authority boundary.
      throw new NotFoundException('Pack not found');
    }
    if (pack.scope !== 'LOCAL') {
      throw new BadRequestException('Only local Packs have a coarse locality');
    }
    if (pack.regionKey === regionKey) {
      return this.repairReceipt(pack.id, regionKey);
    }
    if (pack.regionKey !== null) {
      throw new BadRequestException(
        'This Pack already has an approved locality. Changing an established locality requires a separate reviewed policy.'
      );
    }

    const updated = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      UPDATE dogos_social.packs
      SET region_key = ${regionKey}
      WHERE id = ${packId}
        AND owner_user_id = ${userId}
        AND scope = 'LOCAL'
        AND region_key IS NULL
      RETURNING id
    `);
    if (!updated[0]) {
      throw new BadRequestException('Pack locality changed concurrently; refresh before retrying');
    }
    return this.repairReceipt(packId, regionKey);
  }

  private repairReceipt(packId: string, regionKey: PackCoarseRegionId) {
    const coarseRegion = this.regionById(regionKey);
    if (!coarseRegion) throw new BadRequestException('Choose an approved broad locality');
    return {
      packId,
      localityStatus: 'APPROVED' as const,
      coarseRegion,
      locationContract: PACK_LOCALITY_CONTRACT,
    };
  }

  private regionById(value: string | null) {
    if (value === null) return null;
    return PACK_COARSE_REGIONS.find((region) => region.id === value) ?? null;
  }

  private isApprovedRegionId(value: string | null): value is PackCoarseRegionId {
    return this.regionById(value) !== null;
  }
}
