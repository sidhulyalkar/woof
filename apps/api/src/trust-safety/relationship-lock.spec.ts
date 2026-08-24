import { relationshipLockKey, relationshipLockKeys } from './relationship-lock';

describe('relationship lock keys', () => {
  it('is symmetric for the same unordered user pair', () => {
    expect(relationshipLockKey('user-b', 'user-a')).toBe(relationshipLockKey('user-a', 'user-b'));
  });

  it('deduplicates and sorts multi-party relationship locks deterministically', () => {
    expect(relationshipLockKeys('user-a', ['user-c', 'user-b', 'user-c', 'user-a'])).toEqual([
      relationshipLockKey('user-a', 'user-b'),
      relationshipLockKey('user-a', 'user-c'),
    ]);
  });
});
