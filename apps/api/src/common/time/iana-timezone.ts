const MAX_TIMEZONE_LENGTH = 80;

export function canonicalIanaTimeZone(value: string): string {
  const timezone = value.trim();
  if (!timezone || timezone.length > MAX_TIMEZONE_LENGTH) {
    throw new RangeError('A valid IANA timezone is required');
  }

  try {
    return new Intl.DateTimeFormat('en-US', { timeZone: timezone }).resolvedOptions().timeZone;
  } catch {
    throw new RangeError('A valid IANA timezone is required');
  }
}

export function localDateInTimeZone(instant: Date | string, timezone: string): string {
  const date = instant instanceof Date ? new Date(instant.getTime()) : new Date(instant);
  if (!Number.isFinite(date.getTime())) {
    throw new RangeError('A valid instant is required');
  }

  const canonicalTimezone = canonicalIanaTimeZone(timezone);
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: canonicalTimezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);

  const year = parts.find((part) => part.type === 'year')?.value;
  const month = parts.find((part) => part.type === 'month')?.value;
  const day = parts.find((part) => part.type === 'day')?.value;
  if (!year || !month || !day) {
    throw new RangeError('Unable to derive a local calendar date');
  }

  return `${year}-${month}-${day}`;
}
