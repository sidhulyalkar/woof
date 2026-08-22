import { BadRequestException } from '@nestjs/common';
import type { IngestTrackerObservationDto } from './dto/autopilot.dto';
import type { AutopilotProvider, NormalizedTrackerObservation } from './autopilot.types';

const LOCATION_KEYS = new Set([
  'lat',
  'latitude',
  'lng',
  'lon',
  'longitude',
  'coordinates',
  'position',
  'positions',
  'route',
  'track',
  'trace',
  'gps',
]);

function containsLocationTelemetry(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some((entry) => containsLocationTelemetry(entry));

  return Object.entries(value as Record<string, unknown>).some(
    ([key, nested]) => LOCATION_KEYS.has(key.toLowerCase()) || containsLocationTelemetry(nested)
  );
}

function finiteNumber(value: unknown, label: string): number | undefined {
  if (value === undefined || value === null) return undefined;
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new BadRequestException(`${label} must be a non-negative finite number`);
  }
  return number;
}

function batteryPercent(value: unknown): number | undefined {
  const number = finiteNumber(value, 'batteryPercent');
  if (number === undefined) return undefined;
  if (number > 100) throw new BadRequestException('batteryPercent must be between 0 and 100');
  return number;
}

function deviceState(value: unknown): 'ONLINE' | 'OFFLINE' | 'UNKNOWN' | undefined {
  if (value === undefined || value === null) return undefined;
  const state = String(value).toUpperCase();
  if (state === 'ONLINE' || state === 'ACTIVE' || state === 'CONNECTED') return 'ONLINE';
  if (state === 'OFFLINE' || state === 'INACTIVE' || state === 'DISCONNECTED') return 'OFFLINE';
  return 'UNKNOWN';
}

function requireUsefulMetrics(metrics: NormalizedTrackerObservation['metrics']) {
  if (Object.values(metrics).every((value) => value === undefined)) {
    throw new BadRequestException('Tracker observation did not contain supported metrics');
  }
}

function normalizeFi(dto: IngestTrackerObservationDto): NormalizedTrackerObservation {
  const payload = dto.payload;
  const metrics =
    dto.kind === 'DAILY_ACTIVITY'
      ? {
          activityMinutes: finiteNumber(
            payload.activityMinutes ?? payload.minutesActive ?? payload.activeMinutes,
            'activityMinutes'
          ),
          distanceMeters: finiteNumber(
            payload.distanceMeters ?? payload.distance_m,
            'distanceMeters'
          ),
          steps: finiteNumber(payload.steps, 'steps'),
        }
      : {
          batteryPercent: batteryPercent(payload.batteryPercent ?? payload.battery),
          deviceState: deviceState(payload.deviceState ?? payload.status),
        };
  requireUsefulMetrics(metrics);

  return {
    provider: 'FI',
    externalEventId: dto.externalEventId,
    kind: dto.kind,
    observedAt: new Date(dto.observedAt),
    metrics,
  };
}

function normalizeTractive(dto: IngestTrackerObservationDto): NormalizedTrackerObservation {
  const payload = dto.payload;
  const metrics =
    dto.kind === 'DAILY_ACTIVITY'
      ? {
          activityMinutes: finiteNumber(
            payload.activeMinutes ?? payload.activityMinutes ?? payload.minutes_active,
            'activityMinutes'
          ),
          distanceMeters: finiteNumber(
            payload.distanceMeters ?? payload.distance,
            'distanceMeters'
          ),
          steps: finiteNumber(payload.steps, 'steps'),
        }
      : {
          batteryPercent: batteryPercent(payload.batteryPercent ?? payload.battery_level),
          deviceState: deviceState(payload.deviceState ?? payload.tracker_state ?? payload.status),
        };
  requireUsefulMetrics(metrics);

  return {
    provider: 'TRACTIVE',
    externalEventId: dto.externalEventId,
    kind: dto.kind,
    observedAt: new Date(dto.observedAt),
    metrics,
  };
}

export function normalizeProviderObservation(
  provider: string,
  dto: IngestTrackerObservationDto
): NormalizedTrackerObservation {
  if (containsLocationTelemetry(dto.payload)) {
    throw new BadRequestException(
      'Autopilot v1 does not accept location telemetry; location connectors require explicit retention controls'
    );
  }

  const normalizedProvider = provider.toUpperCase() as AutopilotProvider;
  if (normalizedProvider === 'FI') return normalizeFi(dto);
  if (normalizedProvider === 'TRACTIVE') return normalizeTractive(dto);
  throw new BadRequestException('Unsupported Autopilot provider');
}
