import { apiClient } from './client';

export type MediaTag = {
  label: string;
  source: 'owner' | 'behavior' | 'health' | 'coach' | 'system';
  confidence?: number;
};

export type MediaAsset = {
  id: string;
  createdAt: string;
  filename: string;
  mimeType: string;
  mediaType: 'image' | 'video';
  sizeBytes: number;
  capturedAt: string | null;
  source: string;
  provider: string | null;
  favorite: boolean;
  albumIds: string[];
  smartAlbumIds: string[];
  tags: MediaTag[];
  linkedObservationIds: string[];
  url: string | null;
  urlExpiresInSeconds: number | null;
  status: string;
};

export type MediaAlbum = {
  id: string;
  name: string;
  description: string | null;
  icon: string;
  kind: 'SMART' | 'USER';
  count: number;
  coverAssetId?: string | null;
};

export type MediaLibraryResponse = {
  petId: string;
  assets: MediaAsset[];
  albums: MediaAlbum[];
  storage: { usedBytes: number; quotaBytes: number; storageConfigured: boolean };
  importCapabilities: {
    devicePicker: boolean;
    appleSystemPicker: boolean;
    googlePhotosPicker: boolean;
    googlePhotosBroadLibrarySync: boolean;
  };
};

export type UploadMediaInput = {
  petId: string;
  media: File | Blob;
  filename?: string;
  capturedAt?: string;
  source?:
    | 'camera'
    | 'device-picker'
    | 'google-photos-picker'
    | 'apple-photos-picker'
    | 'health-lens'
    | 'behavior-vision'
    | 'coach'
    | 'imported-file';
  albumIds?: string[];
  tags?: string[];
  linkedObservationId?: string;
};

function filenameFor(media: File | Blob, provided?: string) {
  if (provided?.trim()) return provided.trim();
  if (media instanceof File && media.name) return media.name;
  const ext = media.type.startsWith('video/') ? 'webm' : media.type === 'image/png' ? 'png' : 'jpg';
  return `pet-media-${Date.now()}.${ext}`;
}

async function optionalSha256(media: Blob) {
  // Hashing a very large video in the browser adds unnecessary memory pressure. S3 object
  // verification remains authoritative; the client hash is a useful extra integrity receipt.
  if (media.size > 25 * 1024 * 1024 || !globalThis.crypto?.subtle) return undefined;
  const bytes = await media.arrayBuffer();
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

export const mediaLibraryApi = {
  library: async (petId: string, options?: { albumId?: string; tag?: string; limit?: number }) =>
    apiClient.get('/media-library', {
      params: { petId, ...options },
    }) as unknown as Promise<MediaLibraryResponse>,

  uploadMedia: async (input: UploadMediaInput) => {
    const filename = filenameFor(input.media, input.filename);
    const intent = (await apiClient.post('/media-library/uploads/intents', {
      petId: input.petId,
      filename,
      mimeType: input.media.type || 'application/octet-stream',
      sizeBytes: input.media.size,
      capturedAt: input.capturedAt,
      source: input.source ?? 'device-picker',
      albumIds: input.albumIds ?? [],
      tags: input.tags ?? [],
      linkedObservationId: input.linkedObservationId,
    })) as unknown as {
      assetId: string;
      uploadUrl: string;
      requiredHeaders: Record<string, string>;
    };

    const upload = await fetch(intent.uploadUrl, {
      method: 'PUT',
      headers: intent.requiredHeaders,
      body: input.media,
    });
    if (!upload.ok) throw new Error(`Private media upload failed (${upload.status})`);

    return apiClient.post('/media-library/uploads/complete', {
      assetId: intent.assetId,
      sha256: await optionalSha256(input.media),
    }) as unknown as Promise<MediaAsset>;
  },

  createAlbum: async (petId: string, name: string, description?: string) =>
    apiClient.post('/media-library/albums', {
      petId,
      name,
      description,
    }) as unknown as Promise<MediaAlbum>,

  updateAsset: async (
    assetId: string,
    update: { favorite?: boolean; albumIds?: string[]; tags?: string[] },
  ) => apiClient.patch(`/media-library/assets/${assetId}`, update) as unknown as Promise<MediaAsset>,

  deleteAsset: async (assetId: string) =>
    apiClient.delete(`/media-library/assets/${assetId}`) as unknown as Promise<{
      deleted: boolean;
      assetId: string;
    }>,

  startGooglePhotosPicker: async (petId: string, accessToken: string, maxItemCount = 50) =>
    apiClient.post('/media-library/providers/google-photos/picker', {
      petId,
      accessToken,
      maxItemCount,
    }) as unknown as Promise<{
      sessionId: string;
      pickerUri: string;
      pollingConfig: { pollInterval?: string; timeoutIn?: string } | null;
    }>,

  importGooglePhotos: async (
    petId: string,
    accessToken: string,
    sessionId: string,
    albumIds: string[] = [],
  ) =>
    apiClient.post('/media-library/providers/google-photos/import', {
      petId,
      accessToken,
      sessionId,
      albumIds,
    }) as unknown as Promise<{
      ready: boolean;
      imported: MediaAsset[];
      pollingConfig?: { pollInterval?: string; timeoutIn?: string } | null;
    }>,

  exportGooglePhotos: async (petId: string, accessToken: string, assetIds: string[]) =>
    apiClient.post('/media-library/providers/google-photos/export', {
      petId,
      accessToken,
      assetIds,
    }) as unknown as Promise<{ provider: string; requested: number; uploaded: number }>,

  exportManifest: async (petId: string, assetIds?: string[]) =>
    apiClient.post('/media-library/export/manifest', {
      petId,
      assetIds,
    }) as unknown as Promise<{
      schemaVersion: string;
      generatedAt: string;
      assets: Array<MediaAsset & { downloadUrl: string }>;
    }>,
};
