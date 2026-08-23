import { BadRequestException } from '@nestjs/common';
import type { ConnectorProvider, VerifiedWearableTransportEvent } from './connectors.types';
import { parseConnectorProvider } from './provider-registry';

export const DEVICE_PARTNER_SCHEMA_VERSION = 'woof-device-partner-v1' as const;
export const DEVICE_PARTNER_MAX_PAYLOAD_BYTES = 16 * 1024;
export const DEVICE_PARTNER_MAX_AGE_DAYS = 35;
export const DEVICE_PARTNER_MAX_FUTURE_SKEW_MINUTES = 5;

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

export type DevicePartnerEnvelope = {
  schemaVersion: typeof DEVICE_PARTNER_SCHEMA_VERSION;
  provider: ConnectorProvider;
  externalPetId: string;
  externalObjectId: string;
  kind: 'DAILY_ACTIVITY' | 'DEVICE_STATUS';
  observedAt: string;
  payload: Record<string, unknown>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function boundedIdentifier(value: unknown, label: string) {
  if (typeof value !== 'string') throw new BadRequestException(`${label} must be a string`);
  const normalized = value.trim();
  if (!normalized || normalized.length > 160 || /[\u0000-\u001f\u007f]/.test(normalized)) {
    throw new BadRequestException(
      `${label} must be a non-empty identifier of at most 160 characters`
    );
  }
  return normalized;
}

function containsLocationTelemetry(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some((entry) => containsLocationTelemetry(entry));
  return Object.entries(value as Record<string, unknown>).some(
    ([key, nested]) => LOCATION_KEYS.has(key.toLowerCase()) || containsLocationTelemetry(nested)
  );
}

function parseObservedAt(value: unknown, now: Date) {
  if (typeof value !== 'string')
    throw new BadRequestException('observedAt must be an ISO timestamp');
  const observedAt = new Date(value);
  if (!Number.isFinite(observedAt.getTime())) {
    throw new BadRequestException('observedAt must be a valid timestamp');
  }
  const futureLimit = now.getTime() + DEVICE_PARTNER_MAX_FUTURE_SKEW_MINUTES * 60_000;
  const historyLimit = now.getTime() - DEVICE_PARTNER_MAX_AGE_DAYS * 86_400_000;
  if (observedAt.getTime() > futureLimit) {
    throw new BadRequestException('observedAt is too far in the future');
  }
  if (observedAt.getTime() < historyLimit) {
    throw new BadRequestException('observedAt is outside the supported backfill window');
  }
  return observedAt.toISOString();
}

export function parseDevicePartnerEnvelope(
  input: unknown,
  now = new Date()
): DevicePartnerEnvelope {
  if (!isRecord(input)) throw new BadRequestException('Device partner event must be an object');
  if (input.schemaVersion !== DEVICE_PARTNER_SCHEMA_VERSION) {
    throw new BadRequestException(`schemaVersion must be ${DEVICE_PARTNER_SCHEMA_VERSION}`);
  }

  const provider = parseConnectorProvider(String(input.provider ?? ''));
  if (provider !== 'FI' && provider !== 'TRACTIVE') {
    throw new BadRequestException(
      'Device partner envelopes are supported only for wearable providers'
    );
  }
  if (input.kind !== 'DAILY_ACTIVITY' && input.kind !== 'DEVICE_STATUS') {
    throw new BadRequestException('Unsupported device partner event kind');
  }
  if (!isRecord(input.payload)) throw new BadRequestException('payload must be an object');
  if (containsLocationTelemetry(input.payload)) {
    throw new BadRequestException('Device partner v1 does not accept precise location telemetry');
  }

  let serialized: string;
  try {
    serialized = JSON.stringify(input.payload);
  } catch {
    throw new BadRequestException('payload must be JSON serializable');
  }
  if (Buffer.byteLength(serialized, 'utf8') > DEVICE_PARTNER_MAX_PAYLOAD_BYTES) {
    throw new BadRequestException(
      `payload exceeds the ${DEVICE_PARTNER_MAX_PAYLOAD_BYTES}-byte device contract limit`
    );
  }

  return {
    schemaVersion: DEVICE_PARTNER_SCHEMA_VERSION,
    provider,
    externalPetId: boundedIdentifier(input.externalPetId, 'externalPetId'),
    externalObjectId: boundedIdentifier(input.externalObjectId, 'externalObjectId'),
    kind: input.kind,
    observedAt: parseObservedAt(input.observedAt, now),
    payload: input.payload,
  };
}

export function toVerifiedWearableTransportEvent(
  envelope: DevicePartnerEnvelope
): VerifiedWearableTransportEvent {
  return {
    externalPetId: envelope.externalPetId,
    externalObjectId: envelope.externalObjectId,
    kind: envelope.kind,
    observedAt: envelope.observedAt,
    payload: envelope.payload,
  };
}
