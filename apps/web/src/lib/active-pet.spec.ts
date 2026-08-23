import { beforeEach, describe, expect, it } from 'vitest';
import { clearActivePetId, getActivePetId, setActivePetId } from './active-pet';

describe('active pet context', () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.history.replaceState({}, '', '/');
  });

  it('persists the selected dog in both URL and local storage', () => {
    setActivePetId('pet-2');

    expect(getActivePetId()).toBe('pet-2');
    expect(window.localStorage.getItem('woof.activePetId')).toBe('pet-2');
    expect(new URLSearchParams(window.location.search).get('pet')).toBe('pet-2');
  });

  it('prefers explicit URL context over an older remembered dog', () => {
    window.localStorage.setItem('woof.activePetId', 'pet-old');
    window.history.replaceState({}, '', '/?pet=pet-current');

    expect(getActivePetId()).toBe('pet-current');
  });

  it('clears both durable and URL context', () => {
    setActivePetId('pet-3');
    clearActivePetId();

    expect(getActivePetId()).toBeNull();
    expect(window.localStorage.getItem('woof.activePetId')).toBeNull();
    expect(window.location.search).toBe('');
  });
});
