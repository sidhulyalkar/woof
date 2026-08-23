import { DiscoveryService } from './discovery.service';

describe('DiscoveryService', () => {
  function build() {
    const prisma = {
      pet: { findFirst: jest.fn(), findMany: jest.fn() },
      blockedUser: { findMany: jest.fn() },
      telemetry: { create: jest.fn().mockResolvedValue({}) },
    };
    const locations = {
      upsert: jest.fn(),
      disable: jest.fn(),
      get: jest.fn(),
      findNearby: jest.fn(),
    };
    return {
      prisma,
      locations,
      service: new DiscoveryService(prisma as never, locations as never),
    };
  }

  it('quantizes a precise opt-in coordinate before persistence and never records it in telemetry', async () => {
    const { prisma, locations, service } = build();
    locations.upsert.mockImplementation(async (input: Record<string, unknown>) => ({
      user_id: input.userId,
      lat_bucket: input.latBucket,
      lng_bucket: input.lngBucket,
      precision_m: input.precisionM,
      enabled: true,
      expires_at: input.expiresAt,
      updated_at: new Date('2026-08-23T08:20:00.000Z'),
    }));

    const result = await service.updateLocation('user-1', 37.7749295, -122.4194155);

    expect(locations.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: 'user-1',
        latBucket: expect.any(Number),
        lngBucket: expect.any(Number),
        precisionM: 2200,
      }),
    );
    const persistedInput = locations.upsert.mock.calls[0][0] as Record<string, unknown>;
    expect(persistedInput).not.toHaveProperty('latitude');
    expect(persistedInput).not.toHaveProperty('longitude');
    expect(prisma.telemetry.create).toHaveBeenCalledWith({
      data: {
        userId: 'user-1',
        source: 'discovery',
        event: 'DISCOVERY_LOCATION_OPTED_IN',
        data: { precisionM: 2200, ttlDays: 30 },
      },
    });
    expect(result.exactLocationStored).toBe(false);
  });

  it('returns no discovery candidates when location consent is disabled', async () => {
    const { prisma, locations, service } = build();
    prisma.pet.findFirst.mockResolvedValue({ id: 'pet-1', species: 'DOG' });
    locations.get.mockResolvedValue({
      user_id: 'user-1',
      lat_bucket: 6388,
      lng_bucket: 2879,
      precision_m: 2200,
      enabled: false,
      expires_at: new Date('2026-09-01T00:00:00.000Z'),
      updated_at: new Date(),
    });

    await expect(service.getNearbyCandidates('user-1', 'pet-1')).resolves.toMatchObject({
      locationStatus: 'DISABLED',
      candidates: [],
    });
    expect(locations.findNearby).not.toHaveBeenCalled();
  });

  it('filters blocked members before the public pet candidate query', async () => {
    const { prisma, locations, service } = build();
    prisma.pet.findFirst.mockResolvedValue({ id: 'pet-1', species: 'DOG' });
    locations.get.mockResolvedValue({
      user_id: 'user-1',
      lat_bucket: 6388,
      lng_bucket: 2879,
      precision_m: 2200,
      enabled: true,
      expires_at: new Date(Date.now() + 86_400_000),
      updated_at: new Date(),
    });
    locations.findNearby.mockResolvedValue([
      {
        user_id: 'user-2',
        lat_bucket: 6388,
        lng_bucket: 2880,
        precision_m: 2200,
        enabled: true,
        expires_at: new Date(Date.now() + 86_400_000),
        updated_at: new Date(),
      },
      {
        user_id: 'user-3',
        lat_bucket: 6389,
        lng_bucket: 2880,
        precision_m: 2200,
        enabled: true,
        expires_at: new Date(Date.now() + 86_400_000),
        updated_at: new Date(),
      },
    ]);
    prisma.blockedUser.findMany.mockResolvedValue([
      { userId: 'user-1', blockedId: 'user-2' },
    ]);
    prisma.pet.findMany.mockResolvedValue([
      {
        id: 'pet-3',
        ownerId: 'user-3',
        name: 'Luna',
        species: 'DOG',
        breed: null,
        avatarUrl: null,
        owner: { id: 'user-3', handle: 'luna-human', avatarUrl: null, isVerified: false },
      },
    ]);

    const result = await service.getNearbyCandidates('user-1', 'pet-1', 5, 20);

    expect(prisma.pet.findMany).toHaveBeenCalledWith(
      expect.objectContaining({
        where: expect.objectContaining({
          ownerId: { in: ['user-3'] },
          owner: { visibility: 'PUBLIC' },
        }),
      }),
    );
    expect(result.candidates).toHaveLength(1);
    expect(result.candidates[0]).toEqual(
      expect.objectContaining({ petId: 'pet-3', ownerId: 'user-3', distanceBand: expect.any(String) }),
    );
    expect(result.candidates[0]).not.toHaveProperty('latitude');
    expect(result.candidates[0]).not.toHaveProperty('longitude');
    expect(result.boundaries).toMatchObject({
      exactCoordinatesReturned: false,
      blockedUsersExcluded: true,
      publicProfilesOnly: true,
    });
  });
});
