import { randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';
import { AccountDeletionService } from './account-deletion.service';

describe('AccountDeletionService integration', () => {
  const prisma = new PrismaService();
  const deletedStorageKeys: string[] = [];
  const storage = {
    deleteFile: jest.fn(async (key: string) => {
      deletedStorageKeys.push(key);
    }),
  } as unknown as StorageService;
  const service = new AccountDeletionService(prisma, storage);

  beforeAll(async () => {
    await prisma.$connect();
  });

  beforeEach(() => {
    deletedStorageKeys.length = 0;
    jest.clearAllMocks();
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  async function userFixture(label: string) {
    const suffix = randomUUID().slice(0, 8);
    return prisma.user.create({
      data: {
        handle: `delete-${label}-${suffix}`,
        email: `delete-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
  }

  it('removes private media, legacy identifiers, modern dogOS rows, and empty containers', async () => {
    const [user, peer] = await Promise.all([userFixture('owner'), userFixture('peer')]);
    const pet = await prisma.pet.create({
      data: { ownerId: user.id, name: 'Delete Me Dog', species: 'DOG' },
      select: { id: true },
    });
    const household = await prisma.household.create({
      data: {
        name: 'Deletion household',
        timezone: 'America/Los_Angeles',
        members: { create: { userId: user.id, role: 'OWNER' } },
        pets: { create: { petId: pet.id } },
      },
      select: { id: true },
    });
    const conversation = await prisma.conversation.create({
      data: { participants: { create: { userId: user.id } } },
      select: { id: true },
    });

    const assetKey = `private/library/${randomUUID()}.jpg`;
    const derivativeKey = `private/derivatives/${randomUUID()}.jpg`;
    await prisma.mediaAsset.create({
      data: {
        ownerId: user.id,
        petId: pet.id,
        storageKey: assetKey,
        filename: 'delete-me.jpg',
        mimeType: 'image/jpeg',
        mediaType: 'IMAGE',
        sizeBytes: BigInt(128),
        source: 'IOS',
        status: 'READY',
        createdFrom: 'TEST',
        derivatives: {
          create: {
            kind: 'THUMBNAIL',
            processorVersion: 'test-v1',
            storageKey: derivativeKey,
            mimeType: 'image/jpeg',
            sizeBytes: BigInt(32),
            status: 'READY',
          },
        },
      },
    });

    await prisma.telemetry.create({
      data: { source: 'MOBILE', event: 'DELETE_FIXTURE', userId: user.id, petId: pet.id },
    });
    await prisma.meetupProposal.create({
      data: {
        proposerId: user.id,
        recipientId: peer.id,
        suggestedTime: new Date(),
        suggestedVenue: { name: 'Fixture park', lat: 0, lng: 0 },
        feedbackTags: [],
      },
    });
    await prisma.coActivitySegment.create({
      data: {
        userId: user.id,
        petId: pet.id,
        otherUserId: peer.id,
        startTime: new Date(Date.now() - 60_000),
        endTime: new Date(),
        distanceM: 100,
      },
    });

    const business = await prisma.business.create({
      data: { name: `Deletion business ${randomUUID()}`, type: 'trainer', lat: 0, lng: 0 },
      select: { id: true },
    });
    await prisma.serviceIntent.create({
      data: { userId: user.id, businessId: business.id, action: 'view' },
    });
    await prisma.gamification.create({ data: { userId: user.id, badges: [] } });
    await prisma.pointTransaction.create({
      data: { userId: user.id, points: 1, reason: 'deletion_fixture' },
    });
    await prisma.badgeAward.create({
      data: { userId: user.id, badgeType: `deletion-${randomUUID()}` },
    });
    await prisma.weeklyStreak.create({
      data: { userId: user.id, lastActivityAt: new Date() },
    });
    await prisma.proactiveNudge.create({
      data: {
        userId: user.id,
        targetUserId: peer.id,
        type: 'feedback_request',
        payload: { fixture: true },
      },
    });
    await prisma.nudgeCooldown.create({
      data: {
        userId: peer.id,
        targetUserId: user.id,
        nudgeType: `deletion-${randomUUID()}`,
        cooldownUntil: new Date(Date.now() + 60_000),
      },
    });
    await prisma.safetyVerification.create({
      data: { userId: user.id, verifiedBy: user.id },
    });
    await prisma.reportFlag.create({
      data: {
        reporterId: user.id,
        reportedId: peer.id,
        reason: 'deletion_fixture',
        evidence: [],
        reviewedBy: user.id,
      },
    });
    await prisma.blockedUser.create({
      data: { userId: user.id, blockedId: peer.id },
    });

    const reward = await prisma.reward.create({
      data: {
        code: `delete-${randomUUID()}`,
        title: 'Deletion fixture reward',
        points: 1,
        redeemedBy: user.id,
        redeemedAt: new Date(),
      },
      select: { id: true },
    });
    await prisma.meetup.create({
      data: {
        title: 'Deletion fixture meetup',
        location: { type: 'Point', coordinates: [0, 0] },
        startsAt: new Date(Date.now() + 60_000),
        creatorUserId: user.id,
      },
    });
    await prisma.communityEvent.create({
      data: {
        title: 'Deletion fixture event',
        hostUserId: user.id,
        venueType: 'park',
        lat: 0,
        lng: 0,
        startTime: new Date(Date.now() + 60_000),
        endTime: new Date(Date.now() + 120_000),
      },
    });
    await prisma.mLTrainingData.create({
      data: {
        dataPoint: {
          userFeatures: { userId: user.id, petId: pet.id, features: {} },
          candidateFeatures: { userId: peer.id, petId: 'peer-pet', features: {} },
          timestamp: new Date().toISOString(),
        },
      },
    });

    await prisma.$executeRaw`
      INSERT INTO dogos_auth.sessions (id, user_id, expires_at)
      VALUES (${`delete-session-${randomUUID()}`}, ${user.id}, NOW() + INTERVAL '1 hour')
    `;
    await prisma.$executeRaw`
      INSERT INTO dogos_companion.profiles (user_id, mode)
      VALUES (${user.id}, 'PET_GUARDIAN')
    `;

    await expect(service.deleteCurrentAccount(user.id)).resolves.toBeUndefined();

    expect(deletedStorageKeys.sort()).toEqual([assetKey, derivativeKey].sort());
    expect(storage.deleteFile).toHaveBeenCalledTimes(2);

    await expect(prisma.user.findUnique({ where: { id: user.id } })).resolves.toBeNull();
    await expect(prisma.pet.findUnique({ where: { id: pet.id } })).resolves.toBeNull();
    await expect(prisma.mediaAsset.count({ where: { ownerId: user.id } })).resolves.toBe(0);
    await expect(prisma.telemetry.count({ where: { userId: user.id } })).resolves.toBe(0);
    await expect(
      prisma.meetupProposal.count({
        where: { OR: [{ proposerId: user.id }, { recipientId: user.id }] },
      }),
    ).resolves.toBe(0);
    await expect(
      prisma.coActivitySegment.count({
        where: { OR: [{ userId: user.id }, { otherUserId: user.id }] },
      }),
    ).resolves.toBe(0);
    await expect(prisma.pointTransaction.count({ where: { userId: user.id } })).resolves.toBe(0);
    await expect(prisma.proactiveNudge.count({ where: { userId: user.id } })).resolves.toBe(0);
    await expect(prisma.blockedUser.count({ where: { userId: user.id } })).resolves.toBe(0);
    await expect(prisma.household.findUnique({ where: { id: household.id } })).resolves.toBeNull();
    await expect(
      prisma.conversation.findUnique({ where: { id: conversation.id } }),
    ).resolves.toBeNull();

    const detachedReward = await prisma.reward.findUnique({ where: { id: reward.id } });
    expect(detachedReward?.redeemedBy).toBeNull();
    expect(detachedReward?.redeemedAt).toBeNull();

    const mlRows = await prisma.mLTrainingData.findMany({
      where: {
        OR: [
          { dataPoint: { path: ['userFeatures', 'userId'], equals: user.id } },
          { dataPoint: { path: ['userFeatures', 'petId'], equals: pet.id } },
        ],
      },
    });
    expect(mlRows).toHaveLength(0);

    const sessions = await prisma.$queryRaw<Array<{ id: string }>>`
      SELECT id FROM dogos_auth.sessions WHERE user_id = ${user.id}
    `;
    const companionProfiles = await prisma.$queryRaw<Array<{ user_id: string }>>`
      SELECT user_id FROM dogos_companion.profiles WHERE user_id = ${user.id}
    `;
    expect(sessions).toHaveLength(0);
    expect(companionProfiles).toHaveLength(0);

    await expect(prisma.user.findUnique({ where: { id: peer.id } })).resolves.toMatchObject({
      id: peer.id,
    });
  });

  it('fails closed before relational deletion when private media removal fails', async () => {
    const user = await userFixture('storage-failure');
    const pet = await prisma.pet.create({
      data: { ownerId: user.id, name: 'Storage Failure Dog', species: 'DOG' },
      select: { id: true },
    });
    const asset = await prisma.mediaAsset.create({
      data: {
        ownerId: user.id,
        petId: pet.id,
        storageKey: `private/library/${randomUUID()}.jpg`,
        filename: 'keep-metadata.jpg',
        mimeType: 'image/jpeg',
        mediaType: 'IMAGE',
        sizeBytes: BigInt(64),
        source: 'IOS',
        status: 'READY',
        createdFrom: 'TEST',
      },
      select: { id: true },
    });

    (storage.deleteFile as jest.Mock).mockRejectedValueOnce(new Error('provider unavailable'));

    await expect(service.deleteCurrentAccount(user.id)).rejects.toThrow(
      'Account deletion could not remove private media',
    );
    await expect(prisma.user.findUnique({ where: { id: user.id } })).resolves.toMatchObject({
      id: user.id,
    });
    await expect(prisma.mediaAsset.findUnique({ where: { id: asset.id } })).resolves.toMatchObject({
      id: asset.id,
    });

    await prisma.mediaAsset.delete({ where: { id: asset.id } });
    await prisma.user.delete({ where: { id: user.id } });
  });

  it('deletes a zero-pet account and its embedded legacy ML identity', async () => {
    const [user, peer] = await Promise.all([userFixture('zero-pet'), userFixture('zero-pet-peer')]);
    await prisma.mLTrainingData.create({
      data: {
        dataPoint: {
          userFeatures: { userId: user.id, features: {} },
          candidateFeatures: { userId: peer.id, features: {} },
          timestamp: new Date().toISOString(),
        },
      },
    });

    await expect(service.deleteCurrentAccount(user.id)).resolves.toBeUndefined();
    await expect(prisma.user.findUnique({ where: { id: user.id } })).resolves.toBeNull();
    expect(storage.deleteFile).not.toHaveBeenCalled();

    const rows = await prisma.mLTrainingData.findMany({
      where: { dataPoint: { path: ['userFeatures', 'userId'], equals: user.id } },
    });
    expect(rows).toHaveLength(0);
  });
});
