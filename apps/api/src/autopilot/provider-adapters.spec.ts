import { BadRequestException } from '@nestjs/common';
import { normalizeProviderObservation } from './provider-adapters';

describe('Autopilot provider adapters', () => {
  it('normalizes a Fi daily activity summary without retaining provider-shaped payload data', () => {
    const result = normalizeProviderObservation('fi', {
      petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      externalEventId: 'fi-day-1',
      kind: 'DAILY_ACTIVITY',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: {
        minutesActive: 76,
        distance_m: 5200,
        steps: 9400,
        vendorOnlyField: 'discard-me',
      },
    });

    expect(result).toEqual({
      provider: 'FI',
      externalEventId: 'fi-day-1',
      kind: 'DAILY_ACTIVITY',
      observedAt: new Date('2026-08-22T08:00:00.000Z'),
      metrics: {
        activityMinutes: 76,
        distanceMeters: 5200,
        steps: 9400,
      },
    });
  });

  it('normalizes a Tractive device summary', () => {
    const result = normalizeProviderObservation('tractive', {
      petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      externalEventId: 'tractive-device-1',
      kind: 'DEVICE_STATUS',
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { battery_level: 12, tracker_state: 'connected' },
    });

    expect(result.metrics).toEqual({ batteryPercent: 12, deviceState: 'ONLINE' });
  });

  it('rejects nested location telemetry instead of silently retaining it', () => {
    expect(() =>
      normalizeProviderObservation('fi', {
        petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        externalEventId: 'fi-location-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: {
          activityMinutes: 42,
          vendor: { latest: { coordinates: [-122.1, 37.4] } },
        },
      })
    ).toThrow(BadRequestException);
  });

  it('rejects unsupported providers', () => {
    expect(() =>
      normalizeProviderObservation('mystery-tracker', {
        petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        externalEventId: 'unknown-1',
        kind: 'DEVICE_STATUS',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { batteryPercent: 40 },
      })
    ).toThrow(BadRequestException);
  });

  it('rejects invalid battery percentages', () => {
    expect(() =>
      normalizeProviderObservation('tractive', {
        petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        externalEventId: 'tractive-bad-battery',
        kind: 'DEVICE_STATUS',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { batteryPercent: 140 },
      })
    ).toThrow(BadRequestException);
  });
});
