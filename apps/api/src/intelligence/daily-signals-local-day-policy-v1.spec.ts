import {
  DAILY_SIGNALS_LOCAL_DAY_POLICY_V1,
  dailySignalsDedupeKey,
  resolveDailySignalsTime,
} from './daily-signals-local-day-policy-v1';

describe('daily-signals-local-day-v1', () => {
  const now = new Date('2026-08-26T18:00:00.000Z');

  it('pins the server-authoritative household clock policy', () => {
    expect(DAILY_SIGNALS_LOCAL_DAY_POLICY_V1).toEqual({
      version: 'daily-signals-local-day-v1',
      dedupePrefix: 'daily-signals-v1',
      futureTimestampPolicy: 'CLAMP_TO_SERVER_NOW',
      timezoneAuthority: 'HOUSEHOLD_IANA',
    });
  });

  it('preserves an offline observation on its historical household-local day', () => {
    expect(
      resolveDailySignalsTime({
        requestedObservedAt: '2026-08-25T18:00:00.000Z',
        householdTimezone: 'America/Los_Angeles',
        now,
      })
    ).toEqual({
      observedAt: '2026-08-25T18:00:00.000Z',
      localDate: '2026-08-25',
      timezone: 'America/Los_Angeles',
      futureTimestampNormalized: false,
    });
  });

  it('clamps a future client timestamp before deriving the local day', () => {
    expect(
      resolveDailySignalsTime({
        requestedObservedAt: '2026-09-01T00:00:00.000Z',
        householdTimezone: 'America/Los_Angeles',
        now,
      })
    ).toEqual({
      observedAt: '2026-08-26T18:00:00.000Z',
      localDate: '2026-08-26',
      timezone: 'America/Los_Angeles',
      futureTimestampNormalized: true,
    });
  });

  it('derives the default observation from server now rather than the client', () => {
    expect(
      resolveDailySignalsTime({
        householdTimezone: 'Asia/Tokyo',
        now: new Date('2026-08-25T15:00:00.000Z'),
      })
    ).toEqual({
      observedAt: '2026-08-25T15:00:00.000Z',
      localDate: '2026-08-26',
      timezone: 'Asia/Tokyo',
      futureTimestampNormalized: false,
    });
  });

  it('keeps the same logical date through both fall-back repeated hours', () => {
    const first = resolveDailySignalsTime({
      requestedObservedAt: '2026-11-01T08:30:00.000Z',
      householdTimezone: 'America/Los_Angeles',
      now: new Date('2026-11-02T00:00:00.000Z'),
    });
    const second = resolveDailySignalsTime({
      requestedObservedAt: '2026-11-01T09:30:00.000Z',
      householdTimezone: 'America/Los_Angeles',
      now: new Date('2026-11-02T00:00:00.000Z'),
    });
    expect(first.localDate).toBe('2026-11-01');
    expect(second.localDate).toBe('2026-11-01');
  });

  it('makes actor-independent pet + household + local-day identities', () => {
    expect(
      dailySignalsDedupeKey({
        householdId: '11111111-1111-1111-1111-111111111111',
        petId: '22222222-2222-2222-2222-222222222222',
        localDate: '2026-08-25',
      })
    ).toBe(
      'daily-signals-v1:11111111-1111-1111-1111-111111111111:22222222-2222-2222-2222-222222222222:2026-08-25'
    );
  });

  it('rejects invalid household clocks and malformed observedAt inputs', () => {
    expect(() =>
      resolveDailySignalsTime({
        householdTimezone: 'Mars/Olympus',
        now,
      })
    ).toThrow('valid IANA timezone');
    expect(() =>
      resolveDailySignalsTime({
        requestedObservedAt: 'yesterday-ish',
        householdTimezone: 'America/Los_Angeles',
        now,
      })
    ).toThrow('valid ISO timestamp');
  });
});
