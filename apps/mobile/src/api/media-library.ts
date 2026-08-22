import apiClient from './client';

export type NativeMediaUploadIntent = {
  assetId: string;
  uploadUrl: string;
  requiredHeaders: Record<string, string>;
  expiresIn: number;
};

export type NativeMediaAsset = {
  id: string;
  petId?: string;
  filename: string;
  mimeType: string;
  mediaType: 'image' | 'video';
  sizeBytes: number;
  url: string | null;
  thumbnailUrl?: string | null;
  posterUrl?: string | null;
  previewUrl?: string | null;
};

export const nativeMediaLibraryApi = {
  createUploadIntent: (input: {
    petId: string;
    filename: string;
    mimeType: string;
    sizeBytes: number;
    capturedAt?: string;
    source: 'apple-photos-picker' | 'camera' | 'imported-file';
    tags?: string[];
    linkedObservationId?: string;
  }) => apiClient.post<NativeMediaUploadIntent>('/media-library/uploads/intents', input),

  completeUpload: (assetId: string) =>
    apiClient.post<NativeMediaAsset>('/media-library/uploads/complete', { assetId }),

  library: (petId: string) =>
    apiClient.get<{
      petId: string;
      assets: NativeMediaAsset[];
      storage: { usedBytes: number; quotaBytes: number; storageConfigured: boolean };
    }>('/media-library', { params: { petId, limit: 80 } }),
};
