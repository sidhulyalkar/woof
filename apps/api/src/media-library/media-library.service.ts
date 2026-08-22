import {
  BadRequestException,
  ForbiddenException,
  Injectable,
  NotFoundException,
  ServiceUnavailableException,
} from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';
import {
  CompleteMediaUploadDto,
  CreateMediaAlbumDto,
  CreateMediaUploadIntentDto,
  GooglePhotosExportDto,
  GooglePhotosPickerImportDto,
  GooglePhotosPickerStartDto,
  MediaExportManifestDto,
  MediaLibraryQueryDto,
  UpdateMediaAssetDto,
} from './dto/media-library.dto';
import {
  MEDIA_EXPORT_EVENT,
  MEDIA_IMPORT_EVENT,
  MEDIA_LIBRARY_SOURCE,
  SYSTEM_ALBUMS,
  normalizeMediaTags,
  type MediaAssetData,
  type MediaSource,
  type MediaTag,
} from './media-library.types';

const GOOGLE_PICKER_BASE = 'https://photospicker.googleapis.com/v1';
const GOOGLE_LIBRARY_BASE = 'https://photoslibrary.googleapis.com/v1';
const DEFAULT_IMAGE_MAX = 25 * 1024 * 1024;
const DEFAULT_VIDEO_MAX = 500 * 1024 * 1024;
const DEFAULT_USER_QUOTA = 10 * 1024 * 1024 * 1024;
const MAX_LIBRARY_SCAN = 5000;

const ALLOWED_MIME_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/heic',
  'image/heif',
  'video/mp4',
  'video/webm',
  'video/quicktime',
]);

type AssetRecord = {
  id: string;
  ownerId: string;
  petId: string;
  storageKey: string;
  filename: string;
  mimeType: string;
  mediaType: string;
  sizeBytes: bigint;
  capturedAt: Date | null;
  source: string;
  provider: string | null;
  providerItemId: string | null;
  favorite: boolean;
  status: string;
  createdFrom: string;
  sha256: string | null;
  width: number | null;
  height: number | null;
  durationMs: number | null;
  uploadExpiresAt: Date | null;
  completedAt: Date | null;
  tags: Prisma.JsonValue;
  linkedObservationIds: string[];
  createdAt: Date;
  updatedAt: Date;
  albumLinks?: Array<{ albumId: string }>;
};

@Injectable()
export class MediaLibraryService {
  private readonly imageMaxBytes: number;
  private readonly videoMaxBytes: number;
  private readonly userQuotaBytes: number;

  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
    private readonly config: ConfigService,
  ) {
    this.imageMaxBytes = this.boundedNumber(
      this.config.get('MEDIA_LIBRARY_IMAGE_MAX_BYTES'),
      DEFAULT_IMAGE_MAX,
      1 * 1024 * 1024,
      100 * 1024 * 1024,
    );
    this.videoMaxBytes = this.boundedNumber(
      this.config.get('MEDIA_LIBRARY_VIDEO_MAX_BYTES'),
      DEFAULT_VIDEO_MAX,
      10 * 1024 * 1024,
      1024 * 1024 * 1024,
    );
    this.userQuotaBytes = this.boundedNumber(
      this.config.get('MEDIA_LIBRARY_USER_QUOTA_BYTES'),
      DEFAULT_USER_QUOTA,
      100 * 1024 * 1024,
      1024 * 1024 * 1024 * 1024,
    );
  }

  async createUploadIntent(userId: string, dto: CreateMediaUploadIntentDto) {
    await this.requireOwnedPet(userId, dto.petId);
    this.assertStorageReady();
    this.assertAllowedMedia(dto.mimeType, dto.sizeBytes);
    await this.enforceQuota(userId, dto.sizeBytes);
    const albumIds = await this.validateAlbumIds(userId, dto.petId, dto.albumIds ?? []);
    const source = dto.source ?? 'device-picker';
    const upload = await this.storage.createPrivateUploadIntent({
      filename: dto.filename,
      folder: `private/media/${userId}/${dto.petId}`,
      contentType: dto.mimeType,
      expectedSizeBytes: dto.sizeBytes,
      expiresIn: 900,
    });

    const asset = await this.prisma.mediaAsset.create({
      data: {
        ownerId: userId,
        petId: dto.petId,
        storageKey: upload.key,
        filename: this.cleanFilename(dto.filename),
        mimeType: dto.mimeType,
        mediaType: dto.mimeType.startsWith('video/') ? 'video' : 'image',
        sizeBytes: BigInt(dto.sizeBytes),
        capturedAt: dto.capturedAt ? new Date(dto.capturedAt) : null,
        source,
        provider:
          source === 'apple-photos-picker'
            ? 'APPLE_PHOTOS'
            : source === 'device-picker'
              ? 'DEVICE'
              : null,
        favorite: false,
        status: 'PENDING',
        createdFrom: 'UPLOAD',
        tags: normalizeMediaTags([
          ...this.sourceTags(source),
          ...(dto.tags ?? []).map((label) => ({ label, source: 'owner' as const })),
        ]) as Prisma.InputJsonValue,
        linkedObservationIds: dto.linkedObservationId ? [dto.linkedObservationId] : [],
        uploadExpiresAt: new Date(Date.now() + upload.expiresIn * 1000),
      },
    });

    if (albumIds.length) {
      await this.prisma.mediaAlbumAsset.createMany({
        data: albumIds.map((albumId) => ({ albumId, assetId: asset.id })),
        skipDuplicates: true,
      });
    }

    return {
      assetId: asset.id,
      uploadUrl: upload.uploadUrl,
      expiresIn: upload.expiresIn,
      requiredHeaders: upload.requiredHeaders,
      privacy: {
        visibility: 'PRIVATE',
        objectKeyExposedToOtherUsers: false,
        note: 'Upload goes directly to private object storage. Completing the upload verifies the stored object before it appears in the library.',
      },
    };
  }

  async completeUpload(userId: string, dto: CompleteMediaUploadDto) {
    const asset = await this.requireAsset(userId, dto.assetId, true);
    if (asset.status === 'READY') return this.decorateAsset(asset);
    if (asset.status !== 'PENDING') throw new BadRequestException('Media upload is not pending');
    if (asset.uploadExpiresAt && asset.uploadExpiresAt.getTime() < Date.now()) {
      await this.storage.deleteFile(asset.storageKey).catch(() => undefined);
      await this.prisma.mediaAsset.update({
        where: { id: asset.id },
        data: { status: 'FAILED', completedAt: new Date() },
      });
      throw new BadRequestException('Media upload intent expired before completion');
    }

    let object;
    try {
      object = await this.storage.headObject(asset.storageKey);
    } catch {
      throw new BadRequestException('Uploaded media object could not be verified');
    }
    const sizeMatches = object.sizeBytes === Number(asset.sizeBytes);
    const contentTypeMatches = !object.contentType || object.contentType === asset.mimeType;
    if (!sizeMatches || !contentTypeMatches) {
      await this.storage.deleteFile(asset.storageKey).catch(() => undefined);
      await this.prisma.mediaAsset.update({
        where: { id: asset.id },
        data: { status: 'FAILED', completedAt: new Date() },
      });
      throw new BadRequestException('Uploaded media did not match the declared size or content type');
    }

    if (dto.sha256 && !/^[a-f0-9]{64}$/i.test(dto.sha256)) {
      throw new BadRequestException('sha256 must be a 64-character hexadecimal digest');
    }

    const ready = await this.prisma.mediaAsset.update({
      where: { id: asset.id },
      data: {
        status: 'READY',
        uploadExpiresAt: null,
        completedAt: new Date(),
        sha256: dto.sha256?.toLowerCase() ?? null,
      },
      include: { albumLinks: { select: { albumId: true } } },
    });
    return this.decorateAsset(ready);
  }

  async library(userId: string, query: MediaLibraryQueryDto) {
    await this.requireOwnedPet(userId, query.petId);
    const limit = Math.max(1, Math.min(100, query.limit ?? 60));
    const candidates = await this.prisma.mediaAsset.findMany({
      where: { ownerId: userId, petId: query.petId, status: 'READY' },
      orderBy: [{ capturedAt: 'desc' }, { createdAt: 'desc' }],
      take: query.albumId?.startsWith('smart:') || query.tag ? MAX_LIBRARY_SCAN : Math.min(300, limit * 4),
      include: { albumLinks: { select: { albumId: true } } },
    });

    const filtered = candidates
      .filter((asset) => this.matchesQuery(asset, query))
      .slice(0, limit);
    const assets = await Promise.all(filtered.map((asset) => this.decorateAsset(asset)));
    const albums = await this.albums(userId, query.petId);
    const storage = await this.storageUsage(userId);

    return {
      petId: query.petId,
      assets,
      albums,
      storage: { ...storage, storageConfigured: this.storage.isConfigured() },
      importCapabilities: {
        devicePicker: true,
        appleSystemPicker: true,
        googlePhotosPicker: true,
        googlePhotosBroadLibrarySync: false,
      },
    };
  }

  async albums(userId: string, petId: string) {
    await this.requireOwnedPet(userId, petId);
    const [custom, assets] = await Promise.all([
      this.prisma.mediaAlbum.findMany({
        where: { ownerId: userId, petId },
        orderBy: { createdAt: 'asc' },
        include: { _count: { select: { assets: true } } },
      }),
      this.prisma.mediaAsset.findMany({
        where: { ownerId: userId, petId, status: 'READY' },
        orderBy: { createdAt: 'desc' },
        take: MAX_LIBRARY_SCAN,
        select: {
          id: true,
          ownerId: true,
          petId: true,
          storageKey: true,
          filename: true,
          mimeType: true,
          mediaType: true,
          sizeBytes: true,
          capturedAt: true,
          source: true,
          provider: true,
          providerItemId: true,
          favorite: true,
          status: true,
          createdFrom: true,
          sha256: true,
          width: true,
          height: true,
          durationMs: true,
          uploadExpiresAt: true,
          completedAt: true,
          tags: true,
          linkedObservationIds: true,
          createdAt: true,
          updatedAt: true,
        },
      }),
    ]);

    const system = SYSTEM_ALBUMS.map((album) => ({
      id: album.id,
      name: album.name,
      description: album.description,
      icon: album.icon,
      kind: 'SMART' as const,
      count: assets.filter((asset) => album.matches(this.toSmartAsset(asset), asset.createdAt)).length,
    }));
    return [
      ...system,
      ...custom.map((album) => ({
        id: album.id,
        name: album.name,
        description: album.description,
        icon: 'folder',
        kind: 'USER' as const,
        count: album._count.assets,
        coverAssetId: album.coverAssetId,
      })),
    ];
  }

  async createAlbum(userId: string, dto: CreateMediaAlbumDto) {
    await this.requireOwnedPet(userId, dto.petId);
    const existing = await this.prisma.mediaAlbum.count({ where: { ownerId: userId, petId: dto.petId } });
    if (existing >= 40) throw new BadRequestException('A pet can have at most 40 custom albums');
    const name = dto.name.trim();
    if (!name) throw new BadRequestException('Album name is required');
    const album = await this.prisma.mediaAlbum.create({
      data: {
        ownerId: userId,
        petId: dto.petId,
        name,
        description: dto.description?.trim() || null,
      },
    });
    return { ...album, count: 0, kind: 'USER' as const, icon: 'folder' };
  }

  async updateAsset(userId: string, assetId: string, dto: UpdateMediaAssetDto) {
    const asset = await this.requireAsset(userId, assetId, true);
    const albumIds = dto.albumIds
      ? await this.validateAlbumIds(userId, asset.petId, dto.albumIds)
      : (asset.albumLinks ?? []).map((link) => link.albumId);
    const existingTags = this.parseTags(asset.tags);
    const systemTags = existingTags.filter((tag) => tag.source !== 'owner');
    const ownerTags = dto.tags
      ? dto.tags.map((label) => ({ label, source: 'owner' as const }))
      : existingTags.filter((tag) => tag.source === 'owner');

    await this.prisma.$transaction([
      this.prisma.mediaAsset.update({
        where: { id: asset.id },
        data: {
          favorite: dto.favorite ?? asset.favorite,
          tags: normalizeMediaTags([...systemTags, ...ownerTags]) as Prisma.InputJsonValue,
        },
      }),
      this.prisma.mediaAlbumAsset.deleteMany({ where: { assetId: asset.id } }),
      ...(albumIds.length
        ? [
            this.prisma.mediaAlbumAsset.createMany({
              data: albumIds.map((albumId) => ({ albumId, assetId: asset.id })),
              skipDuplicates: true,
            }),
          ]
        : []),
    ]);

    return this.decorateAsset(await this.requireAsset(userId, asset.id, true));
  }

  async deleteAsset(userId: string, assetId: string) {
    const asset = await this.requireAsset(userId, assetId);
    await this.storage.deleteFile(asset.storageKey);
    await this.prisma.mediaAsset.delete({ where: { id: asset.id } });
    return { deleted: true, assetId };
  }

  async startGooglePhotosPicker(userId: string, dto: GooglePhotosPickerStartDto) {
    await this.requireOwnedPet(userId, dto.petId);
    const response = await fetch(`${GOOGLE_PICKER_BASE}/sessions`, {
      method: 'POST',
      headers: this.googleHeaders(dto.accessToken),
      body: JSON.stringify({
        pickingConfig: { maxItemCount: String(Math.max(1, Math.min(100, dto.maxItemCount ?? 50))) },
      }),
    });
    const payload = await this.requireGoogleJson(response, 'Could not start Google Photos picker');
    const pickerUri = typeof payload.pickerUri === 'string' ? payload.pickerUri : null;
    const sessionId = typeof payload.id === 'string' ? payload.id : null;
    if (!pickerUri || !sessionId) {
      throw new ServiceUnavailableException('Google Photos returned an invalid picker session');
    }
    return {
      sessionId,
      pickerUri: `${pickerUri.replace(/\/$/, '')}/autoclose`,
      pollingConfig: this.isObject(payload.pollingConfig) ? payload.pollingConfig : null,
      privacy:
        'Woof receives only the photos and videos you explicitly choose in Google Photos. The OAuth token is used for this request and is not persisted.',
    };
  }

  async importGooglePhotos(userId: string, dto: GooglePhotosPickerImportDto) {
    await this.requireOwnedPet(userId, dto.petId);
    this.assertStorageReady();
    const albumIds = await this.validateAlbumIds(userId, dto.petId, dto.albumIds ?? []);
    const sessionResponse = await fetch(
      `${GOOGLE_PICKER_BASE}/sessions/${encodeURIComponent(dto.sessionId)}`,
      { headers: { Authorization: `Bearer ${dto.accessToken}` } },
    );
    const session = await this.requireGoogleJson(
      sessionResponse,
      'Could not read Google Photos picker session',
    );
    if (session.mediaItemsSet !== true) {
      return {
        ready: false,
        imported: [],
        pollingConfig: this.isObject(session.pollingConfig) ? session.pollingConfig : null,
      };
    }

    const selected = await this.listGooglePickedItems(dto.accessToken, dto.sessionId);
    if (selected.length > 50) throw new BadRequestException('Import at most 50 Google Photos items at a time');
    const imported = [];

    for (const item of selected) {
      const mediaFile = this.objectField(item, 'mediaFile');
      const mimeType = this.stringField(mediaFile, 'mimeType');
      const baseUrl = this.stringField(mediaFile, 'baseUrl');
      const providerItemId = this.stringField(item, 'id');
      const filename =
        this.stringField(mediaFile, 'filename') || `google-photo-${providerItemId || Date.now()}`;
      if (!mimeType || !baseUrl || !ALLOWED_MIME_TYPES.has(mimeType)) continue;

      if (providerItemId) {
        const duplicate = await this.prisma.mediaExternalReference.findFirst({
          where: { ownerId: userId, petId: dto.petId, provider: 'GOOGLE_PHOTOS', providerItemId },
          select: { assetId: true },
        });
        if (duplicate) continue;
      }

      const mediaResponse = await fetch(`${baseUrl}=${mimeType.startsWith('video/') ? 'dv' : 'd'}`, {
        headers: { Authorization: `Bearer ${dto.accessToken}` },
      });
      if (!mediaResponse.ok) continue;
      const declared = Number(mediaResponse.headers.get('content-length') || 0);
      const maxBytes = mimeType.startsWith('video/') ? this.videoMaxBytes : this.imageMaxBytes;
      if (declared > maxBytes) continue;

      let stored;
      let sizeBytes: number;
      if (declared > 0 && mediaResponse.body) {
        await this.enforceQuota(userId, declared);
        stored = await this.storage.uploadPrivateWebStream({
          body: mediaResponse.body,
          contentLength: declared,
          filename,
          contentType: mimeType,
          folder: `private/media/${userId}/${dto.petId}`,
        });
        sizeBytes = declared;
      } else {
        const bytes = Buffer.from(await mediaResponse.arrayBuffer());
        this.assertAllowedMedia(mimeType, bytes.byteLength);
        await this.enforceQuota(userId, bytes.byteLength);
        stored = await this.storage.uploadPrivateBytes({
          bytes,
          filename,
          contentType: mimeType,
          folder: `private/media/${userId}/${dto.petId}`,
        });
        sizeBytes = bytes.byteLength;
      }

      const capturedAtValue = this.stringField(item, 'createTime');
      const asset = await this.prisma.mediaAsset.create({
        data: {
          ownerId: userId,
          petId: dto.petId,
          storageKey: stored.key,
          filename: this.cleanFilename(filename),
          mimeType,
          mediaType: mimeType.startsWith('video/') ? 'video' : 'image',
          sizeBytes: BigInt(sizeBytes),
          capturedAt: capturedAtValue ? new Date(capturedAtValue) : null,
          source: 'google-photos-picker',
          provider: 'GOOGLE_PHOTOS',
          providerItemId: providerItemId || null,
          status: 'READY',
          createdFrom: 'IMPORT',
          tags: normalizeMediaTags([
            { label: 'imported', source: 'system' },
            { label: 'google photos', source: 'system' },
          ]) as Prisma.InputJsonValue,
          completedAt: new Date(),
        },
      });
      if (albumIds.length) {
        await this.prisma.mediaAlbumAsset.createMany({
          data: albumIds.map((albumId) => ({ albumId, assetId: asset.id })),
          skipDuplicates: true,
        });
      }
      if (providerItemId) {
        await this.prisma.mediaExternalReference.create({
          data: {
            assetId: asset.id,
            ownerId: userId,
            petId: dto.petId,
            provider: 'GOOGLE_PHOTOS',
            providerItemId,
          },
        });
      }
      imported.push(await this.decorateAsset(await this.requireAsset(userId, asset.id, true)));
    }

    await fetch(`${GOOGLE_PICKER_BASE}/sessions/${encodeURIComponent(dto.sessionId)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${dto.accessToken}` },
    }).catch(() => undefined);
    await this.recordProviderEvent(userId, dto.petId, MEDIA_IMPORT_EVENT, {
      provider: 'GOOGLE_PHOTOS',
      importedCount: imported.length,
    });
    return { ready: true, imported };
  }

  async exportToGooglePhotos(userId: string, dto: GooglePhotosExportDto) {
    await this.requireOwnedPet(userId, dto.petId);
    if (!dto.assetIds.length) throw new BadRequestException('Choose at least one media asset');
    const entries = await this.prisma.mediaAsset.findMany({
      where: {
        id: { in: [...new Set(dto.assetIds)] },
        ownerId: userId,
        petId: dto.petId,
        status: 'READY',
      },
    });
    if (entries.length !== new Set(dto.assetIds).size) {
      throw new ForbiddenException('One or more media assets are unavailable');
    }

    const uploadTokens: Array<{ uploadToken: string; fileName: string }> = [];
    for (const asset of entries) {
      const bytes = await this.storage.getObjectBytes(asset.storageKey, this.videoMaxBytes);
      const response = await fetch(`${GOOGLE_LIBRARY_BASE}/uploads`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${dto.accessToken}`,
          'Content-Type': 'application/octet-stream',
          'X-Goog-Upload-Content-Type': asset.mimeType,
          'X-Goog-Upload-Protocol': 'raw',
        },
        body: new Uint8Array(bytes),
      });
      if (!response.ok) continue;
      const uploadToken = (await response.text()).trim();
      if (uploadToken) uploadTokens.push({ uploadToken, fileName: asset.filename });
    }
    if (!uploadTokens.length) {
      throw new ServiceUnavailableException('No selected media could be uploaded to Google Photos');
    }

    const createResponse = await fetch(`${GOOGLE_LIBRARY_BASE}/mediaItems:batchCreate`, {
      method: 'POST',
      headers: this.googleHeaders(dto.accessToken),
      body: JSON.stringify({
        newMediaItems: uploadTokens.map((item) => ({
          simpleMediaItem: { uploadToken: item.uploadToken, fileName: item.fileName },
        })),
      }),
    });
    const result = await this.requireGoogleJson(createResponse, 'Could not finish Google Photos export');
    await this.recordProviderEvent(userId, dto.petId, MEDIA_EXPORT_EVENT, {
      provider: 'GOOGLE_PHOTOS',
      requestedCount: dto.assetIds.length,
      uploadedCount: uploadTokens.length,
    });
    return {
      provider: 'GOOGLE_PHOTOS',
      requested: dto.assetIds.length,
      uploaded: uploadTokens.length,
      results: Array.isArray(result.newMediaItemResults) ? result.newMediaItemResults : [],
      privacy: 'Woof does not persist the Google OAuth access token used for this export.',
    };
  }

  async exportManifest(userId: string, dto: MediaExportManifestDto) {
    const pet = await this.requireOwnedPet(userId, dto.petId);
    const entries = await this.prisma.mediaAsset.findMany({
      where: {
        ...(dto.assetIds?.length ? { id: { in: dto.assetIds } } : {}),
        ownerId: userId,
        petId: dto.petId,
        status: 'READY',
      },
      orderBy: { createdAt: 'asc' },
      take: 100,
      include: { albumLinks: { select: { albumId: true } } },
    });
    const assets = await Promise.all(
      entries.map(async (asset) => ({
        id: asset.id,
        filename: asset.filename,
        mimeType: asset.mimeType,
        sizeBytes: Number(asset.sizeBytes),
        capturedAt: asset.capturedAt?.toISOString() ?? null,
        createdAt: asset.createdAt.toISOString(),
        source: asset.source,
        favorite: asset.favorite,
        tags: this.parseTags(asset.tags),
        albumIds: asset.albumLinks.map((link) => link.albumId),
        linkedObservationIds: asset.linkedObservationIds,
        downloadUrl: await this.storage.getSignedUrl(asset.storageKey, 900),
        downloadUrlExpiresInSeconds: 900,
      })),
    );
    return {
      schemaVersion: 'woof-media-export-v1',
      generatedAt: new Date().toISOString(),
      pet: { id: pet.id, name: pet.name, species: pet.species },
      assets,
      portability:
        'Download URLs are intentionally short-lived. Asset metadata remains portable and does not expose Woof object-storage keys.',
    };
  }

  private matchesQuery(asset: AssetRecord, query: MediaLibraryQueryDto) {
    const tags = this.parseTags(asset.tags);
    if (query.tag) {
      const tag = query.tag.trim().toLowerCase();
      if (!tags.some((entry) => entry.label.toLowerCase() === tag)) return false;
    }
    if (!query.albumId) return true;
    if (query.albumId.startsWith('smart:')) {
      const smart = SYSTEM_ALBUMS.find((album) => album.id === query.albumId);
      return smart ? smart.matches(this.toSmartAsset(asset), asset.createdAt) : false;
    }
    return (asset.albumLinks ?? []).some((link) => link.albumId === query.albumId);
  }

  private async decorateAsset(asset: AssetRecord) {
    const url =
      asset.status === 'READY'
        ? await this.storage.getSignedUrl(asset.storageKey, 900).catch(() => null)
        : null;
    const tags = this.parseTags(asset.tags);
    const smart = this.toSmartAsset(asset);
    return {
      id: asset.id,
      createdAt: asset.createdAt.toISOString(),
      filename: asset.filename,
      mimeType: asset.mimeType,
      mediaType: asset.mediaType === 'video' ? 'video' : 'image',
      sizeBytes: Number(asset.sizeBytes),
      capturedAt: asset.capturedAt?.toISOString() ?? null,
      source: asset.source,
      provider: asset.provider,
      favorite: asset.favorite,
      albumIds: (asset.albumLinks ?? []).map((link) => link.albumId),
      smartAlbumIds: SYSTEM_ALBUMS.filter((album) => album.matches(smart, asset.createdAt)).map(
        (album) => album.id,
      ),
      tags,
      linkedObservationIds: asset.linkedObservationIds,
      url,
      urlExpiresInSeconds: url ? 900 : null,
      status: asset.status,
    };
  }

  private toSmartAsset(asset: AssetRecord): MediaAssetData {
    return {
      schemaVersion: 'woof-media-asset-v1',
      status: asset.status as MediaAssetData['status'],
      storageKey: asset.storageKey,
      filename: asset.filename,
      mimeType: asset.mimeType,
      sizeBytes: Number(asset.sizeBytes),
      capturedAt: asset.capturedAt?.toISOString() ?? null,
      source: asset.source as MediaAssetData['source'],
      provider: asset.provider as MediaAssetData['provider'],
      providerItemId: asset.providerItemId,
      favorite: asset.favorite,
      albumIds: (asset.albumLinks ?? []).map((link) => link.albumId),
      tags: this.parseTags(asset.tags),
      linkedObservationIds: asset.linkedObservationIds,
      createdFrom: asset.createdFrom as MediaAssetData['createdFrom'],
      uploadExpiresAt: asset.uploadExpiresAt?.toISOString() ?? null,
      completedAt: asset.completedAt?.toISOString() ?? null,
      sha256: asset.sha256,
      width: asset.width,
      height: asset.height,
      durationMs: asset.durationMs,
    };
  }

  private async validateAlbumIds(userId: string, petId: string, albumIds: string[]) {
    const unique = [...new Set(albumIds.filter((id) => !id.startsWith('smart:')))].slice(0, 12);
    if (!unique.length) return [];
    const count = await this.prisma.mediaAlbum.count({
      where: { id: { in: unique }, ownerId: userId, petId },
    });
    if (count !== unique.length) throw new ForbiddenException('One or more albums are unavailable');
    return unique;
  }

  private async storageUsage(userId: string) {
    const aggregate = await this.prisma.mediaAsset.aggregate({
      where: { ownerId: userId, status: { in: ['PENDING', 'READY'] } },
      _sum: { sizeBytes: true },
    });
    return {
      usedBytes: Number(aggregate._sum.sizeBytes ?? 0n),
      quotaBytes: this.userQuotaBytes,
    };
  }

  private async enforceQuota(userId: string, incomingBytes: number) {
    const usage = await this.storageUsage(userId);
    if (usage.usedBytes + incomingBytes > this.userQuotaBytes) {
      throw new BadRequestException('This upload would exceed the private media storage quota');
    }
  }

  private assertAllowedMedia(mimeType: string, sizeBytes: number) {
    if (!ALLOWED_MIME_TYPES.has(mimeType)) throw new BadRequestException('Unsupported media type');
    const max = mimeType.startsWith('video/') ? this.videoMaxBytes : this.imageMaxBytes;
    if (sizeBytes > max) {
      throw new BadRequestException(
        mimeType.startsWith('video/')
          ? 'Video exceeds the media-library size limit'
          : 'Image exceeds the media-library size limit',
      );
    }
  }

  private assertStorageReady() {
    if (!this.storage.isConfigured()) {
      throw new ServiceUnavailableException('Private media storage is not configured');
    }
  }

  private sourceTags(source: MediaSource): MediaTag[] {
    if (source === 'behavior-vision') return [{ label: 'behavior', source: 'behavior' }];
    if (source === 'health-lens') return [{ label: 'health', source: 'health' }];
    if (source === 'coach') return [{ label: 'training', source: 'coach' }];
    return [];
  }

  private parseTags(value: Prisma.JsonValue): MediaTag[] {
    if (!Array.isArray(value)) return [];
    const tags: MediaTag[] = [];
    for (const item of value) {
      if (!this.isObject(item)) continue;
      const label = this.stringField(item, 'label');
      const source = this.stringField(item, 'source');
      if (!label || !['owner', 'behavior', 'health', 'coach', 'system'].includes(source)) continue;
      const confidenceValue = item.confidence;
      tags.push({
        label,
        source: source as MediaTag['source'],
        confidence:
          typeof confidenceValue === 'number'
            ? Math.max(0, Math.min(1, confidenceValue))
            : undefined,
      });
    }
    return normalizeMediaTags(tags);
  }

  private async listGooglePickedItems(accessToken: string, sessionId: string) {
    const items: Array<Record<string, unknown>> = [];
    let pageToken: string | null = null;
    do {
      const url = new URL(`${GOOGLE_PICKER_BASE}/mediaItems`);
      url.searchParams.set('sessionId', sessionId);
      url.searchParams.set('pageSize', '100');
      if (pageToken) url.searchParams.set('pageToken', pageToken);
      const response = await fetch(url, { headers: { Authorization: `Bearer ${accessToken}` } });
      const payload = await this.requireGoogleJson(
        response,
        'Could not list selected Google Photos media',
      );
      const pageItems = Array.isArray(payload.mediaItems) ? payload.mediaItems : [];
      items.push(...pageItems.filter((item): item is Record<string, unknown> => this.isObject(item)));
      pageToken = this.stringField(payload, 'nextPageToken') || null;
    } while (pageToken && items.length < 100);
    return items.slice(0, 100);
  }

  private googleHeaders(accessToken: string) {
    return { Authorization: `Bearer ${accessToken}`, 'Content-Type': 'application/json' };
  }

  private async requireGoogleJson(response: Response, message: string): Promise<Record<string, unknown>> {
    const text = await response.text();
    if (!response.ok) throw new ServiceUnavailableException(`${message} (${response.status})`);
    try {
      const parsed: unknown = text ? JSON.parse(text) : {};
      return this.isObject(parsed) ? parsed : {};
    } catch {
      throw new ServiceUnavailableException(`${message}: invalid provider response`);
    }
  }

  private async requireAsset(userId: string, assetId: string, includeAlbums = false) {
    const asset = await this.prisma.mediaAsset.findFirst({
      where: { id: assetId, ownerId: userId },
      include: includeAlbums ? { albumLinks: { select: { albumId: true } } } : undefined,
    });
    if (!asset) throw new NotFoundException('Media asset not found');
    return asset as AssetRecord;
  }

  private async requireOwnedPet(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true, name: true, species: true },
    });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');
    return pet;
  }

  private async recordProviderEvent(
    userId: string,
    petId: string,
    event: string,
    data: Record<string, Prisma.InputJsonValue>,
  ) {
    await this.prisma.telemetry.create({
      data: {
        source: MEDIA_LIBRARY_SOURCE,
        event,
        userId,
        petId,
        data: { ...data, occurredAt: new Date().toISOString() } as Prisma.InputJsonValue,
      },
    });
  }

  private objectField(value: Record<string, unknown>, key: string) {
    const candidate = value[key];
    return this.isObject(candidate) ? candidate : {};
  }

  private stringField(value: Record<string, unknown>, key: string) {
    const candidate = value[key];
    return typeof candidate === 'string' ? candidate : '';
  }

  private isObject(value: unknown): value is Record<string, unknown> {
    return Boolean(value) && !Array.isArray(value) && typeof value === 'object';
  }

  private cleanFilename(filename: string) {
    const cleaned = filename.replace(/[\\/\0\r\n]/g, '_').trim();
    return (cleaned || 'pet-media').slice(0, 240);
  }

  private boundedNumber(value: unknown, fallback: number, min: number, max: number) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.max(min, Math.min(max, Math.round(parsed)));
  }
}
