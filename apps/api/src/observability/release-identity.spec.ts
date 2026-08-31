import {
  UNKNOWN_RELEASE,
  resolveProcessReleaseIdentity,
  resolveReleaseIdentity,
} from './release-identity';

describe('release identity authority', () => {
  const originalReleaseSha = process.env.WOOF_RELEASE_SHA;

  afterEach(() => {
    if (originalReleaseSha === undefined) {
      delete process.env.WOOF_RELEASE_SHA;
    } else {
      process.env.WOOF_RELEASE_SHA = originalReleaseSha;
    }
  });

  it('accepts and normalizes one exact Git SHA', () => {
    expect(resolveReleaseIdentity('ABCDEF0123456789ABCDEF0123456789ABCDEF01')).toBe(
      'abcdef0123456789abcdef0123456789abcdef01'
    );
  });

  it.each([
    undefined,
    '',
    'main',
    'latest',
    'abcdef',
    'abcdef0123456789abcdef0123456789abcdef0z',
    'abcdef0123456789abcdef0123456789abcdef0123',
  ])('fails visibly closed for non-SHA candidate identity %#', (value) => {
    expect(resolveReleaseIdentity(value)).toBe(UNKNOWN_RELEASE);
  });

  it('reads process deployment identity only through the explicit process resolver', () => {
    process.env.WOOF_RELEASE_SHA = 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA';
    expect(resolveProcessReleaseIdentity()).toBe('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
  });

  it('does not let ambient process identity replace an explicitly invalid candidate', () => {
    process.env.WOOF_RELEASE_SHA = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
    expect(resolveReleaseIdentity(undefined)).toBe(UNKNOWN_RELEASE);
    expect(resolveReleaseIdentity('main')).toBe(UNKNOWN_RELEASE);
  });
});
