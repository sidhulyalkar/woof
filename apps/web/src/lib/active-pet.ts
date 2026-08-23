const ACTIVE_PET_STORAGE_KEY = 'woof.activePetId';
const ACTIVE_PET_QUERY_KEY = 'pet';

export function getActivePetId() {
  if (typeof window === 'undefined') return null;

  const fromUrl = new URLSearchParams(window.location.search).get(ACTIVE_PET_QUERY_KEY);
  if (fromUrl) return fromUrl;

  return window.localStorage.getItem(ACTIVE_PET_STORAGE_KEY);
}

export function setActivePetId(petId: string) {
  if (typeof window === 'undefined') return;

  window.localStorage.setItem(ACTIVE_PET_STORAGE_KEY, petId);
  const url = new URL(window.location.href);
  url.searchParams.set(ACTIVE_PET_QUERY_KEY, petId);
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
}

export function clearActivePetId() {
  if (typeof window === 'undefined') return;

  window.localStorage.removeItem(ACTIVE_PET_STORAGE_KEY);
  const url = new URL(window.location.href);
  url.searchParams.delete(ACTIVE_PET_QUERY_KEY);
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
}
