import { BadRequestException } from '@nestjs/common';
import {
  DEVICE_PARTNER_MAX_PAYLOAD_BYTES,
  parseDevicePartnerEnvelope,
} from './device-partner-contract';

const now = new Date('2026-08-23T21:00:00.000Z');

function envelope(overrides: Record<string, unknown> = {}) {
  return {
    schemaVersion: 'woof-device-partner-v1',
    provider: 'TRACTIVE',
    externalPetId: 'pet-external-1',
    externalObjectId: 'activity-1',
    kind: 'DAILY_ACTIVITY',
    observedAt: '2026-08-23T20:00:00.000Z',
    payload: { activeMinutes: 42, distance: 3500 },
    ...overrides,
  };
}

describe('device partner contract', () => {
  it('accepts a bounded versioned wearable event and normalizes its timestamp', () => {
    expect(parseDevicePartnerEnvelope(envelope(), now)).toEqual(
      expect.objectContaining({
        schemaVersion: 'woof-device-partner-v1',
        provider: 'TRACTIVE',
        externalPetId: 'pet-external-1',
        externalObjectId: 'activity-1',
        observedAt: '2026-08-23T20:00:00.000Z',
      })
    );
  });

  it('rejects precise location telemetry anywhere in the payload', () => {
    expect(() =>
      parseDevicePartnerEnvelope(
        envelope({ payload: { activeMinutes: 42, nested: { coordinates: [1, 2] } } }),
        now
      )
    ).toThrow(BadRequestException);
  });

  it('rejects stale, future-skewed, or unversioned events', () => {
    expect(() =>
      parseDevicePartnerEnvelope(envelope({ observedAt: '2026-06-01T00:00:00.000Z' }), now)
    ).toThrow(BadRequestException);
    expect(() =>
      parseDevicePartnerEnvelope(envelope({ observedAt: '2026-08-23T21:06:00.000Z' }), now)
    ).toThrow(BadRequestException);
    expect(() => parseDevicePartnerEnvelope(envelope({ schemaVersion: 'v0' }), now)).toThrow(
      BadRequestException
    );
  });

  it('rejects oversized payloads before they reach provider normalization', () => {
    const oversized = 'x'.repeat(DEVICE_PARTNER_MAX_PAYLOAD_BYTES + 1);
    expect(() =>
      parseDevicePartnerEnvelope(envelope({ payload: { blob: oversized } }), now)
    ).toThrow(BadRequestException);
  });
});
