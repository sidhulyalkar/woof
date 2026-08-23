import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { DiscoveryLocationStore } from './discovery-location.store';

const CELL_DEGREES = 0.02;
const CELL_PRECISION_M = 2_200;
const LOCATION_TTL_DAYS = 30;
const DAY_MS = 24 * 60 * 60 * 1000;

type DistanceBand = 'WITHIN_2_5_KM' | 'WITHIN_5_KM' | 'WITHIN_10_KM';

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function quantizeLatitude(latitude: number) {
  return clamp(Math.floor((latitude + 90) / CELL_DEGREES), 0, 9000);
}

function quantizeLongitude(longitude: number) {
  return clamp(Math.floor((longitude + 180) / CELL_DEGREES), 0, 18000);
}

function bucketCenter(bucket: number, offset: number) {
  return bucket * CELL_DEGREES - offset + CELL_DEGREES / 2;
}

function approximateKm(
  source: { latBucket: number; lngBucket: number },
  target: { latBucket: number; lngBucket: number },
) {
  const sourceLat = bucketCenter(source.latBucket, 90);
  const targetLat = bucketCenter(target.latBucket, 90);
  const latKm = Math.abs(sourceLat - targetLat) * 111.32;
  const meanLatRadians = ((sourceLat + targetLat) / 2) * (Math.PI / 180);
  const sourceLng = bucketCenter(source.lngBucket, 180);
  const targetLng = bucketCenter(target.lngBucket, 180);
  const lngKm = Math.abs(sourceLng - targetLng) * 111.32 * Math.max(Math.cos(meanLatRadians), 0.2);
  return Math.sqrt(latKm * latKm + lngKm * lngKm);
}

function distanceBand(distanceKm: number): DistanceBand {
  if (distanceKm <= 2.5) return 'WITHIN_2_5_KM';
  if (distanceKm <= 5) return 'WITHIN_5_KM';
  return 'WITHIN_10_KM';
}

@Injectable()
export class DiscoveryService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly locations: DiscoveryLocationStore,
  ) {}

  async updateLocation(userId: string, latitude: number, longitude: number) {
    const expiresAt = new Date(Date.now() + LOCATION_TTL_DAYS * DAY_MS);
    const row = await this.locations.upsert({
      userId,
      latBucket: quantizeLatitude(latitude),
      lngBucket: quantizeLongitude(longitude),
      precisionM: CELL_PRECISION_M,
      expiresAt,
    });

    await this.recordTelemetry(userId, 'DISCOVERY_LOCATION_OPTED_IN', {
      precisionM: CELL_PRECISION_M,
      ttlDays: LOCATION_TTL_DAYS,
    });

    return {
      status: 'OPTED_IN' as const,
      precisionMeters: row.precision_m,
      expiresAt: row.expires_at.toISOString(),
      exactLocationStored: false,
    };
  }

  async disableLocation(userId: string) {
    await this.locations.disable(userId);
    await this.recordTelemetry(userId, 'DISCOVERY_LOCATION_DISABLED', {});
    return { status: 'DISABLED' as const, exactLocationStored: false };
  }

  async getStatus(userId: string) {
    const row = await this.locations.get(userId);
    if (!row) {
      return {
        status: 'NOT_CONFIGURED' as const,
        exactLocationStored: false,
        precisionMeters: CELL_PRECISION_M,
      };
    }
    if (!row.enabled) {
      return {
        status: 'DISABLED' as const,
        exactLocationStored: false,
        precisionMeters: row.precision_m,
      };
    }
    if (row.expires_at.getTime() <= Date.now()) {
      return {
        status: 'STALE' as const,
        exactLocationStored: false,
        precisionMeters: row.precision_m,
      };
    }
    return {
      status: 'OPTED_IN' as const,
      exactLocationStored: false,
      precisionMeters: row.precision_m,
      expiresAt: row.expires_at.toISOString(),
      updatedAt: row.updated_at.toISOString(),
    };
  }

  async getNearbyCandidates(userId: string, petId: string, radiusKm = 5, limit = 20) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true, species: true },
    });
    if (!pet) throw new NotFoundException('Pet not found');

    const ownLocation = await this.locations.get(userId);
    if (!ownLocation || !ownLocation.enabled || ownLocation.expires_at.getTime() <= Date.now()) {
      return {
        petId,
        locationStatus: ownLocation?.enabled ? 'STALE' : ownLocation ? 'DISABLED' : 'NOT_CONFIGURED',
        candidates: [],
        boundaries: this.boundaries(),
      };
    }

    const safeRadiusKm = clamp(Number(radiusKm) || 5, 2, 10);
    const safeLimit = clamp(Number(limit) || 20, 1, 30);
    const sourceLatitude = bucketCenter(ownLocation.lat_bucket, 90);
    const latCellKm = 111.32 * CELL_DEGREES;
    const lngCellKm = latCellKm * Math.max(Math.cos(sourceLatitude * (Math.PI / 180)), 0.2);
    const latRange = Math.ceil(safeRadiusKm / latCellKm) + 1;
    const lngRange = Math.ceil(safeRadiusKm / lngCellKm) + 1;

    const nearbyRows = await this.locations.findNearby({
      userId,
      minLatBucket: ownLocation.lat_bucket - latRange,
      maxLatBucket: ownLocation.lat_bucket + latRange,
      minLngBucket: ownLocation.lng_bucket - lngRange,
      maxLngBucket: ownLocation.lng_bucket + lngRange,
      take: 400,
    });

    const nearbyOwnerIds = nearbyRows.map((row) => row.user_id);
    if (nearbyOwnerIds.length === 0) {
      return {
        petId,
        locationStatus: 'OPTED_IN' as const,
        candidates: [],
        boundaries: this.boundaries(),
      };
    }

    const blocks = await this.prisma.blockedUser.findMany({
      where: {
        OR: [
          { userId, blockedId: { in: nearbyOwnerIds } },
          { blockedId: userId, userId: { in: nearbyOwnerIds } },
        ],
      },
      select: { userId: true, blockedId: true },
    });
    const blockedOwnerIds = new Set(
      blocks.map((block) => (block.userId === userId ? block.blockedId : block.userId)),
    );
    const eligibleOwnerIds = nearbyOwnerIds.filter((ownerId) => !blockedOwnerIds.has(ownerId));

    const pets = await this.prisma.pet.findMany({
      where: {
        id: { not: petId },
        ownerId: { in: eligibleOwnerIds },
        species: pet.species,
        owner: { visibility: 'PUBLIC' },
      },
      select: {
        id: true,
        ownerId: true,
        name: true,
        species: true,
        breed: true,
        avatarUrl: true,
        owner: {
          select: { id: true, handle: true, avatarUrl: true, isVerified: true },
        },
      },
      take: 100,
    });

    const locationByOwner = new Map(nearbyRows.map((row) => [row.user_id, row]));
    const candidates = pets
      .map((candidate) => {
        const candidateLocation = locationByOwner.get(candidate.ownerId);
        if (!candidateLocation) return null;
        const approximateDistanceKm = approximateKm(
          { latBucket: ownLocation.lat_bucket, lngBucket: ownLocation.lng_bucket },
          { latBucket: candidateLocation.lat_bucket, lngBucket: candidateLocation.lng_bucket },
        );
        if (approximateDistanceKm > safeRadiusKm + 1.5) return null;
        return {
          petId: candidate.id,
          ownerId: candidate.ownerId,
          petName: candidate.name,
          species: candidate.species,
          breed: candidate.breed,
          avatarUrl: candidate.avatarUrl,
          owner: candidate.owner,
          distanceBand: distanceBand(approximateDistanceKm),
        };
      })
      .filter((candidate): candidate is NonNullable<typeof candidate> => candidate !== null)
      .sort((a, b) => {
        const order: Record<DistanceBand, number> = {
          WITHIN_2_5_KM: 0,
          WITHIN_5_KM: 1,
          WITHIN_10_KM: 2,
        };
        return order[a.distanceBand] - order[b.distanceBand];
      })
      .slice(0, safeLimit);

    await this.recordTelemetry(userId, 'DISCOVERY_CANDIDATES_VIEWED', {
      petId,
      radiusBandKm: safeRadiusKm,
      candidateCount: candidates.length,
    });

    return {
      petId,
      locationStatus: 'OPTED_IN' as const,
      candidates,
      boundaries: this.boundaries(),
    };
  }

  private boundaries() {
    return {
      exactCoordinatesStored: false,
      exactCoordinatesReturned: false,
      homeLocationExposed: false,
      blockedUsersExcluded: true,
      publicProfilesOnly: true,
      maxRadiusKm: 10,
    };
  }

  private async recordTelemetry(userId: string, event: string, data: Prisma.InputJsonObject) {
    await this.prisma.telemetry.create({
      data: { userId, source: 'discovery', event, data },
    });
  }
}
