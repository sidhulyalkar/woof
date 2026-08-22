import { SYSTEM_ALBUMS, normalizeMediaTags, type MediaAssetData } from './media-library.types';

function asset(overrides: Partial<MediaAssetData> = {}): MediaAssetData {
  return {
    schemaVersion: 'woof-media-asset-v1',
    status: 'READY',
    storageKey: 'private/media/u/p/a.jpg',
    filename: 'a.jpg',
    mimeType: 'image/jpeg',
    sizeBytes: 123,
    capturedAt: null,
    source: 'device-picker',
    provider: null,
    favorite: false,
    albumIds: [],
    tags: [],
    linkedObservationIds: [],
    createdFrom: 'UPLOAD',
    ...overrides,
  };
}

describe('media library smart organization', () => {
  it('sorts behavior and health media by provenance rather than filenames', () => {
    const behavior = asset({ source: 'behavior-vision' });
    const health = asset({ source: 'health-lens' });
    const createdAt = new Date();
    expect(SYSTEM_ALBUMS.find((album) => album.id === 'smart:behavior')?.matches(behavior, createdAt)).toBe(true);
    expect(SYSTEM_ALBUMS.find((album) => album.id === 'smart:health')?.matches(health, createdAt)).toBe(true);
    expect(SYSTEM_ALBUMS.find((album) => album.id === 'smart:health')?.matches(behavior, createdAt)).toBe(false);
  });

  it('deduplicates normalized tags without erasing provenance', () => {
    const tags = normalizeMediaTags([
      { label: ' Trail ', source: 'owner' },
      { label: 'trail', source: 'owner' },
      { label: 'trail', source: 'system', confidence: 1.4 },
    ]);
    expect(tags).toEqual([
      { label: 'trail', source: 'owner', confidence: undefined },
      { label: 'trail', source: 'system', confidence: 1 },
    ]);
  });

  it('never treats imports as public media', () => {
    const imported = asset({
      source: 'google-photos-picker',
      provider: 'GOOGLE_PHOTOS',
      createdFrom: 'IMPORT',
    });
    expect(SYSTEM_ALBUMS.find((album) => album.id === 'smart:imports')?.matches(imported, new Date())).toBe(true);
    expect(imported).not.toHaveProperty('visibility', 'PUBLIC');
  });
});
