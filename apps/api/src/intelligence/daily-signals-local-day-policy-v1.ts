import { canonicalIanaTimeZone, localDateInTimeZone } from '../common/time/iana-timezone';

export const DAILY_SIGNALS_LOCAL_DAY_POLICY_V1 = Object.freeze({
  version: 'daily-signals-local-day-v1' as const,
  dedupePrefix: 'daily-signals-v1' as const,
  futureTimestampPolicy: 'CLAMP_TO_SERVER_NOW' as const,
  timezoneAuthority: 'HOUSEHOLD_IANA' as const,
});

export type DailySignalsResolvedTime = {
  observedAt: string;
  localDate: string;
  timezone: string;
  futureTimestampNormalized: boolean;
};

export function resolveDailySignalsTime(input: {
  requestedObservedAt?: string;
  householdTimezone: string;
  now: Date;
}): DailySignalsResolvedTime {
  const nowMs = input.now.getTime();
  if (!Number.isFinite(nowMs)) throw new RangeError('A valid server now is required');

  const timezone = canonicalIanaTimeZone(input.householdTimezone);
  const requestedMs =
    input.requestedObservedAt === undefined ? nowMs : Date.parse(input.requestedObservedAt);
  if (!Number.isFinite(requestedMs)) {
    throw new RangeError('Daily Signals observedAt must be a valid ISO timestamp');
  }

  const futureTimestampNormalized = requestedMs > nowMs;
  const observedAt = new Date(futureTimestampNormalized ? nowMs : requestedMs);
  return {
    observedAt: observedAt.toISOString(),
    localDate: localDateInTimeZone(observedAt, timezone),
    timezone,
    futureTimestampNormalized,
  };
}

export function dailySignalsDedupeKey(input: {
  householdId: string;
  petId: string;
  localDate: string;
}): string {
  if (!input.householdId || !input.petId || !/^\d{4}-\d{2}-\d{2}$/.test(input.localDate)) {
    throw new RangeError('Daily Signals dedupe identity is invalid');
  }
  return `${DAILY_SIGNALS_LOCAL_DAY_POLICY_V1.dedupePrefix}:${input.householdId}:${input.petId}:${input.localDate}`;
}
