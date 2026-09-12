import { BadRequestException, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import { PACK_COARSE_REGIONS } from './pack-locality.catalog';
import { PackLocalityService } from './pack-locality.service';

describe('PackLocalityService integration', () => {
  const prisma = new PrismaService();
  const service = new PackLocalityService(prisma);
  const usersToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (usersToDelete.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function createUser(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `pack-locality-${label}-${suffix}`,
        email: `pack-locality-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user.id;
  }

  async function createLocalPack(ownerUserId: string, regionKey: string | null) {
    const id = randomUUID();
    const slug = `pack-locality-${id.slice(0, 8)}`;
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.packs (
        id,
        owner_user_id,
        name,
        slug,
        scope,
        region_key,
        visibility
      ) VALUES (
        ${id},
        ${ownerUserId},
        'Pack locality integration',
        ${slug},
        'LOCAL',
        ${regionKey},
        'PUBLIC'
      )
    `);
    await prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.pack_memberships (pack_id, user_id, role, status)
      VALUES (${id}, ${ownerUserId}, 'OWNER', 'ACTIVE')
    `);
    return id;
  }

  it('ships only deliberately broad client-selectable regions', () => {
    expect(PACK_COARSE_REGIONS.length).toBeGreaterThan(0);
    expect(PACK_COARSE_REGIONS).toContainEqual(
      expect.objectContaining({
        id: 'us-ca-south-bay',
        granularity: 'BROAD_DISTRICT',
      })
    );
    const clientRegionIds: readonly string[] = PACK_COARSE_REGIONS.map((region) => region.id);
    expect(clientRegionIds).not.toContain('test-region');
  });

  it('lets migrated legacy Pack identity survive but blocks locality-dependent authority', async () => {
    const ownerId = await createUser('legacy-owner');
    const packId = await createLocalPack(ownerId, null);

    await expect(service.requireLocalityAuthority(packId)).rejects.toBeInstanceOf(
      BadRequestException
    );

    const decorated = service.decorateCatalog({
      packs: [
        {
          id: packId,
          name: 'Legacy Pack',
          slug: 'legacy-pack',
          scope: 'LOCAL',
          regionKey: null,
          visibility: 'PUBLIC',
          memberCount: 1,
          joined: true,
          role: 'OWNER',
        },
      ],
      localMinimumCohort: 5,
      locationContract: 'legacy-value-must-not-win',
    });

    expect(decorated.packs).toHaveLength(1);
    expect(decorated.packs[0]).toMatchObject({
      id: packId,
      localityStatus: 'LEGACY_UNVERIFIED',
      coarseRegion: null,
    });
    expect(decorated.locationContract).toContain('server-approved coarse region only');
  });

  it('hides an unverified legacy Pack from nonmember discovery', async () => {
    const ownerId = await createUser('hidden-owner');
    const packId = await createLocalPack(ownerId, null);

    const decorated = service.decorateCatalog({
      packs: [
        {
          id: packId,
          name: 'Hidden legacy Pack',
          slug: 'hidden-legacy-pack',
          scope: 'LOCAL',
          regionKey: null,
          visibility: 'PUBLIC',
          memberCount: 1,
          joined: false,
          role: null,
        },
      ],
      localMinimumCohort: 5,
      locationContract: 'legacy-value-must-not-win',
    });

    expect(decorated.packs).toEqual([]);
  });

  it('allows only the owner to perform one conservative approved-region repair', async () => {
    const ownerId = await createUser('repair-owner');
    const strangerId = await createUser('repair-stranger');
    const packId = await createLocalPack(ownerId, null);

    await expect(
      service.repairLocality(strangerId, packId, 'us-ca-south-bay')
    ).rejects.toBeInstanceOf(NotFoundException);

    await expect(service.repairLocality(ownerId, packId, 'us-ca-south-bay')).resolves.toMatchObject(
      {
        packId,
        localityStatus: 'APPROVED',
        coarseRegion: { id: 'us-ca-south-bay', displayName: 'South Bay, CA' },
      }
    );

    await expect(service.requireLocalityAuthority(packId)).resolves.toBeUndefined();

    await expect(service.repairLocality(ownerId, packId, 'us-ca-south-bay')).resolves.toMatchObject(
      {
        localityStatus: 'APPROVED',
      }
    );

    await expect(service.repairLocality(ownerId, packId, 'us-ca-peninsula')).rejects.toBeInstanceOf(
      BadRequestException
    );
  });

  it('keeps arbitrary locality text outside the database authority boundary', async () => {
    const ownerId = await createUser('fk-owner');
    await expect(createLocalPack(ownerId, '123-main-st-apartment-4')).rejects.toThrow();
  });
});
