import { Platform } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import * as ImagePicker from 'expo-image-picker';
import * as Sharing from 'expo-sharing';
import { nativeMediaLibraryApi, type NativeMediaAsset } from '../api/media-library';

export type ApplePickedPetMedia = {
  uri: string;
  filename: string;
  mimeType: string;
  sizeBytes: number;
  width: number | null;
  height: number | null;
  durationMs: number | null;
  assetId: string | null;
  source: 'apple-photos-picker';
};

function extensionFor(mimeType: string) {
  if (mimeType === 'image/png') return 'png';
  if (mimeType === 'image/heic' || mimeType === 'image/heif') return 'heic';
  if (mimeType === 'video/quicktime') return 'mov';
  if (mimeType.startsWith('video/')) return 'mp4';
  return 'jpg';
}

function safeFilename(value: string | null | undefined, mimeType: string, index: number) {
  const cleaned = value?.replace(/[\\/\0\r\n]/g, '_').trim();
  return (cleaned || `pet-media-${Date.now()}-${index}.${extensionFor(mimeType)}`).slice(0, 220);
}

async function sizeFor(uri: string, reported: number | null | undefined) {
  if (typeof reported === 'number' && reported > 0) return reported;
  const info = await FileSystem.getInfoAsync(uri, { size: true });
  if (!info.exists || typeof info.size !== 'number' || info.size <= 0) {
    throw new Error('Could not determine the selected media size.');
  }
  return info.size;
}

/**
 * Uses Apple's system picker through Expo ImagePicker. It returns only media the
 * person explicitly selects and does not enumerate the full Photos library.
 */
export async function pickApplePetMedia(maxItems = 20): Promise<ApplePickedPetMedia[]> {
  if (Platform.OS !== 'ios') throw new Error('Apple PhotosPicker is available on iOS/iPadOS only.');

  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ['images', 'videos'],
    allowsMultipleSelection: true,
    selectionLimit: Math.max(1, Math.min(20, maxItems)),
    quality: 1,
    base64: false,
    exif: false,
    preferredAssetRepresentationMode: ImagePicker.UIImagePickerPreferredAssetRepresentationMode.Compatible,
  });
  if (result.canceled) return [];

  const normalized: ApplePickedPetMedia[] = [];
  for (const [index, asset] of result.assets.entries()) {
    const mimeType = asset.mimeType || (asset.type === 'video' ? 'video/mp4' : 'image/jpeg');
    normalized.push({
      uri: asset.uri,
      filename: safeFilename(asset.fileName, mimeType, index),
      mimeType,
      sizeBytes: await sizeFor(asset.uri, asset.fileSize),
      width: Number.isFinite(asset.width) ? asset.width : null,
      height: Number.isFinite(asset.height) ? asset.height : null,
      durationMs: Number.isFinite(asset.duration) ? asset.duration ?? null : null,
      assetId: asset.assetId ?? null,
      source: 'apple-photos-picker',
    });
  }
  return normalized;
}

/** Uploads directly from the local iOS file URI to the short-lived private object URL. */
export async function keepAppleMediaPrivately(input: {
  petId: string;
  items: ApplePickedPetMedia[];
  tags?: string[];
  linkedObservationId?: string;
}) {
  const saved: NativeMediaAsset[] = [];
  for (const item of input.items.slice(0, 20)) {
    const intent = await nativeMediaLibraryApi.createUploadIntent({
      petId: input.petId,
      filename: item.filename,
      mimeType: item.mimeType,
      sizeBytes: item.sizeBytes,
      source: 'apple-photos-picker',
      tags: input.tags,
      linkedObservationId: input.linkedObservationId,
    });

    const upload = await FileSystem.uploadAsync(intent.uploadUrl, item.uri, {
      httpMethod: 'PUT',
      uploadType: FileSystem.FileSystemUploadType.BINARY_CONTENT,
      headers: intent.requiredHeaders,
    });
    if (upload.status < 200 || upload.status >= 300) {
      throw new Error(`Private media upload failed (${upload.status}).`);
    }
    saved.push(await nativeMediaLibraryApi.completeUpload(intent.assetId));
  }
  return saved;
}

function utiFor(mimeType: string) {
  if (mimeType === 'image/jpeg') return 'public.jpeg';
  if (mimeType === 'image/png') return 'public.png';
  if (mimeType === 'image/heic' || mimeType === 'image/heif') return 'public.heic';
  if (mimeType === 'video/quicktime') return 'com.apple.quicktime-movie';
  if (mimeType.startsWith('video/')) return 'public.mpeg-4';
  return 'public.data';
}

/** Downloads a short-lived private original to the app cache, opens iOS share sheet, then deletes it. */
export async function sharePrivateMediaOnApple(asset: NativeMediaAsset) {
  if (Platform.OS !== 'ios') throw new Error('Apple share sheet is available on iOS/iPadOS only.');
  if (!asset.url) throw new Error('This media item does not currently have a private download URL.');
  if (!(await Sharing.isAvailableAsync())) throw new Error('Sharing is unavailable on this device.');
  if (!FileSystem.cacheDirectory) throw new Error('Temporary sharing storage is unavailable.');

  const filename = safeFilename(asset.filename, asset.mimeType, 0);
  const target = `${FileSystem.cacheDirectory}woof-share-${Date.now()}-${filename}`;
  try {
    const download = await FileSystem.downloadAsync(asset.url, target);
    if (download.status < 200 || download.status >= 300) {
      throw new Error(`Private media download failed (${download.status}).`);
    }
    await Sharing.shareAsync(download.uri, {
      UTI: utiFor(asset.mimeType),
      mimeType: asset.mimeType,
      dialogTitle: `Share ${asset.filename}`,
    });
  } finally {
    await FileSystem.deleteAsync(target, { idempotent: true }).catch(() => undefined);
  }
}
