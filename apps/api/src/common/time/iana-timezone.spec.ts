import { canonicalIanaTimeZone, localDateInTimeZone } from './iana-timezone';

describe('IANA timezone utilities', () => {
  it('canonicalizes valid zones and rejects invalid zones', () => {
    expect(canonicalIanaTimeZone(' America/Los_Angeles ')).toBe('America/Los_Angeles');
    expect(() => canonicalIanaTimeZone('Mars/Olympus')).toThrow('valid IANA timezone');
    expect(() => canonicalIanaTimeZone('')).toThrow('valid IANA timezone');
  });

  it('derives calendar dates across the spring DST jump', () => {
    expect(localDateInTimeZone('2026-03-08T09:59:59.000Z', 'America/Los_Angeles')).toBe(
      '2026-03-08'
    );
    expect(localDateInTimeZone('2026-03-08T10:00:00.000Z', 'America/Los_Angeles')).toBe(
      '2026-03-08'
    );
  });

  it('keeps both repeated fall-back hours on the same local day', () => {
    expect(localDateInTimeZone('2026-11-01T08:30:00.000Z', 'America/Los_Angeles')).toBe(
      '2026-11-01'
    );
    expect(localDateInTimeZone('2026-11-01T09:30:00.000Z', 'America/Los_Angeles')).toBe(
      '2026-11-01'
    );
  });

  it('changes the date exactly at local midnight', () => {
    expect(localDateInTimeZone('2026-08-25T06:59:59.999Z', 'America/Los_Angeles')).toBe(
      '2026-08-24'
    );
    expect(localDateInTimeZone('2026-08-25T07:00:00.000Z', 'America/Los_Angeles')).toBe(
      '2026-08-25'
    );
  });

  it('works for non-US timezone boundaries', () => {
    expect(localDateInTimeZone('2026-08-25T14:59:59.999Z', 'Asia/Tokyo')).toBe('2026-08-25');
    expect(localDateInTimeZone('2026-08-25T15:00:00.000Z', 'Asia/Tokyo')).toBe('2026-08-26');
  });

  it('rejects invalid instants', () => {
    expect(() => localDateInTimeZone('not-a-date', 'America/Los_Angeles')).toThrow('valid instant');
  });
});
