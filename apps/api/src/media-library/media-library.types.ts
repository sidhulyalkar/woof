export const MEDIA_LIBRARY_SOURCE = 'MEDIA_LIBRARY';
export const MEDIA_ASSET_EVENT = 'MEDIA_ASSET';
export const MEDIA_ALBUM_EVENT = 'MEDIA_ALBUM';
export const MEDIA_EXPORT_EVENT = 'MEDIA_EXPORT';
export const MEDIA_IMPORT_EVENT = 'MEDIA_IMPORT';
export const MEDIA_ASSET_SCHEMA_VERSION = 'woof-media-asset-v1';
export const MEDIA_ALBUM_SCHEMA_VERSION = 'woof-media-album-v1';

export const MEDIA_SOURCES = [
  'camera',
  'device-picker',
  'google-photos-picker',
  'apple-photos-picker',
  'health-lens',
  'behavior-vision',
  'coach',
  'imported-file',
] as const;
export type MediaSource = (typeof MEDIA_SOURCES)[number];

export const MEDIA_TAG_SOURCES = ['owner', 'behavior', 'health', 'coach', 'system'] as const;
export type MediaTagSource = (typeof MEDIA_TAG_SOURCES)[number];

export type MediaTag = {
  label: string;
  source: MediaTagSource;
  confidence?: number;
};

export type MediaAssetData = {
  schemaVersion: typeof MEDIA_ASSET_SCHEMA_VERSION;
  status: 'PENDING' | 'READY' | 'FAILED' | 'DELETED';
  storageKey: string;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  capturedAt: string | null;
  source: MediaSource;
  providerItemId?: string | null;
  provider?: 'GOOGLE_PHOTOS' | 'APPLE_PHOTOS' | 'DEVICE' | null;
  favorite: boolean;
  albumIds: string[];
  tags: MediaTag[];
  linkedObservationIds: string[];
  createdFrom: 'UPLOAD' | 'IMPORT' | 'ANALYSIS_SAVE';
  uploadExpiresAt?: string | null;
  completedAt?: string | null;
  sha256?: string | null;
  width?: number | null;
  height?: number | null;
  durationMs?: number | null;
};

export type MediaAlbumData = {
  schemaVersion: typeof MEDIA_ALBUM_SCHEMA_VERSION;
  name: string;
  description: string | null;
  petId: string;
  kind: 'USER';
  coverAssetId: string | null;
};

export type SystemAlbumDefinition = {
  id: string;
  name: string;
  description: string;
  icon: string;
  matches: (asset: MediaAssetData, createdAt: Date) => boolean;
};

export const SYSTEM_ALBUMS: SystemAlbumDefinition[] = [
  {
    id: 'smart:recent',
    name: 'Recent',
    description: 'Your newest pet moments from the last 30 days.',
    icon: 'clock',
    matches: (_asset, createdAt) => Date.now() - createdAt.getTime() <= 30 * 24 * 60 * 60 * 1000,
  },
  {
    id: 'smart:favorites',
    name: 'Favorites',
    description: 'Moments you marked as especially worth keeping.',
    icon: 'heart',
    matches: (asset) => asset.favorite,
  },
  {
    id: 'smart:behavior',
    name: 'Behavior',
    description: 'Clips and photos connected to Behavior Vision observations.',
    icon: 'brain',
    matches: (asset) =>
      asset.source === 'behavior-vision' ||
      asset.tags.some((tag) => tag.source === 'behavior' || tag.label === 'behavior'),
  },
  {
    id: 'smart:health',
    name: 'Health',
    description: 'Private visual records connected to health observations.',
    icon: 'heart-pulse',
    matches: (asset) =>
      asset.source === 'health-lens' ||
      asset.tags.some((tag) => tag.source === 'health' || tag.label === 'health'),
  },
  {
    id: 'smart:training',
    name: 'Training',
    description: 'Practice clips, progress moments, and cooperative-care wins.',
    icon: 'sparkles',
    matches: (asset) =>
      asset.source === 'coach' ||
      asset.tags.some((tag) => tag.source === 'coach' || tag.label === 'training'),
  },
  {
    id: 'smart:adventures',
    name: 'Adventures',
    description: 'Walks, hikes, parks, trails, and life outside the house.',
    icon: 'map',
    matches: (asset) =>
      asset.tags.some((tag) =>
        ['walk', 'hike', 'trail', 'park', 'adventure', 'travel'].includes(tag.label.toLowerCase()),
      ),
  },
  {
    id: 'smart:imports',
    name: 'Imports',
    description: 'Media selected from your device or another photo service.',
    icon: 'download',
    matches: (asset) => Boolean(asset.provider) || asset.createdFrom === 'IMPORT',
  },
];

export function normalizeMediaTags(tags: MediaTag[]) {
  const seen = new Set<string>();
  return tags
    .map((tag) => ({
      label: tag.label.trim().toLowerCase().slice(0, 48),
      source: tag.source,
      confidence:
        tag.confidence === undefined
          ? undefined
          : Math.max(0, Math.min(1, Number(tag.confidence) || 0)),
    }))
    .filter((tag) => {
      if (!tag.label) return false;
      const key = `${tag.source}:${tag.label}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .slice(0, 24);
}
