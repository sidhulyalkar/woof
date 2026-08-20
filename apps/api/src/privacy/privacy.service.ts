import { ForbiddenException, Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { UpdatePrivacyPreferencesDto } from './dto/update-privacy-preferences.dto';
import {
  DEFAULT_PRIVACY_PREFERENCES,
  PrivacyPreferences,
} from './privacy.types';

const PREFERENCES_EVENT = 'PRIVACY_PREFERENCES_UPDATED';

@Injectable()
export class PrivacyService {
  constructor(private readonly prisma: PrismaService) {}

  async getPreferences(userId: string): Promise<PrivacyPreferences> {
    const latest = await this.prisma.telemetry.findFirst({
      where: { userId, event: PREFERENCES_EVENT },
      orderBy: { createdAt: 'desc' },
      select: { data: true },
    });
    const data = this.asRecord(latest?.data);
    return this.normalizePreferences(data);
  }

  async updatePreferences(userId: string, dto: UpdatePrivacyPreferencesDto) {
    const current = await this.getPreferences(userId);
    const next: PrivacyPreferences = {
      ...current,
      ...dto,
      // Proximity suggestions can never remain enabled when precise location is disabled.
      proximitySuggestions:
        dto.preciseLocation === false
          ? false
          : (dto.proximitySuggestions ?? current.proximitySuggestions),
    };

    const entry = await this.prisma.telemetry.create({
      data: {
        userId,
        source: 'privacy',
        event: PREFERENCES_EVENT,
        data: next,
      },
    });

    await this.pruneLocationHistory(userId, next.locationRetentionHours);
    if (!next.preciseLocation) {
      await this.clearLocationHistory(userId);
    }

    return { preferences: next, updatedAt: entry.createdAt };
  }

  async requirePreciseLocation(userId: string) {
    const preferences = await this.getPreferences(userId);
    if (!preferences.preciseLocation) {
      throw new ForbiddenException(
        'Precise location is off. Enable it explicitly in privacy settings before using location features.',
      );
    }
    return preferences;
  }

  async canUseProximity(userId: string) {
    const preferences = await this.getPreferences(userId);
    return preferences.preciseLocation && preferences.proximitySuggestions;
  }

  async bothAllowProximity(userAId: string, userBId: string) {
    const [a, b, blocked] = await Promise.all([
      this.getPreferences(userAId),
      this.getPreferences(userBId),
      this.prisma.blockedUser.findFirst({
        where: {
          OR: [
            { userId: userAId, blockedId: userBId },
            { userId: userBId, blockedId: userAId },
          ],
        },
        select: { id: true },
      }),
    ]);
    return (
      !blocked &&
      a.preciseLocation &&
      a.proximitySuggestions &&
      b.preciseLocation &&
      b.proximitySuggestions
    );
  }

  async clearLocationHistory(userId: string) {
    const result = await this.prisma.locationPing.deleteMany({ where: { userId } });
    return { deleted: result.count };
  }

  async pruneLocationHistory(userId: string, retentionHours?: number) {
    const hours = Math.max(1, Math.min(retentionHours ?? 12, 24));
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    const result = await this.prisma.locationPing.deleteMany({
      where: { userId, timestamp: { lt: cutoff } },
    });
    return { deleted: result.count, cutoff };
  }

  async getLocationSummary(userId: string) {
    const preferences = await this.getPreferences(userId);
    const [count, newest, oldest] = await Promise.all([
      this.prisma.locationPing.count({ where: { userId } }),
      this.prisma.locationPing.findFirst({
        where: { userId },
        orderBy: { timestamp: 'desc' },
        select: { timestamp: true },
      }),
      this.prisma.locationPing.findFirst({
        where: { userId },
        orderBy: { timestamp: 'asc' },
        select: { timestamp: true },
      }),
    ]);
    return {
      preferences,
      storedLocationPings: count,
      oldestStoredAt: oldest?.timestamp ?? null,
      newestStoredAt: newest?.timestamp ?? null,
      maxRetentionHours: 24,
    };
  }

  private normalizePreferences(value: Record<string, unknown>): PrivacyPreferences {
    const sharing =
      value.meetupLocationSharing === 'NEVER' ||
      value.meetupLocationSharing === 'AFTER_CONFIRMATION'
        ? value.meetupLocationSharing
        : DEFAULT_PRIVACY_PREFERENCES.meetupLocationSharing;
    const retention =
      typeof value.locationRetentionHours === 'number'
        ? Math.max(1, Math.min(Math.round(value.locationRetentionHours), 24))
        : DEFAULT_PRIVACY_PREFERENCES.locationRetentionHours;
    return {
      preciseLocation:
        typeof value.preciseLocation === 'boolean'
          ? value.preciseLocation
          : DEFAULT_PRIVACY_PREFERENCES.preciseLocation,
      proximitySuggestions:
        typeof value.proximitySuggestions === 'boolean'
          ? value.proximitySuggestions
          : DEFAULT_PRIVACY_PREFERENCES.proximitySuggestions,
      shareActivityRoutes:
        typeof value.shareActivityRoutes === 'boolean'
          ? value.shareActivityRoutes
          : DEFAULT_PRIVACY_PREFERENCES.shareActivityRoutes,
      meetupLocationSharing: sharing,
      locationRetentionHours: retention,
    };
  }

  private asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  }
}
