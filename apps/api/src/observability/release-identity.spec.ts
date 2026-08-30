import { UNKNOWN_RELEASE, resolveReleaseIdentity } from './release-identity';

describe('release identity authority', () => {
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
  ])('fails visibly closed for non-SHA deployment identity %#', (value) => {
    expect(resolveReleaseIdentity(value)).toBe(UNKNOWN_RELEASE);
  });
});
