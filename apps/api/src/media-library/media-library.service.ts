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
  MEDIA_ALBUM_EVENT,
  MEDIA_ALBUM_SCHEMA_VERSION,
  MEDIA_ASSET_EVENT,
  MEDIA_ASSET_SCHEMA_VERSION,
  MEDIA_EXPORT_EVENT,
  MEDIA_IMPORT_EVENT,
  MEDIA_LIBRARY_SOURCE,
  SYSTEM_ALBUMS,
  normalizeMediaTags,
  type MediaAlbumData,
  type MediaAssetData,
  type MediaSource,
} from './media-library.types';

const GOOGLE_PICKER_BASE = 'https://photospicker.googleapis.com/v1';
const GOOGLE_LIBRARY_BASE = 'https://photoslibrary.googleapis.com/v1';
const DEFAULT_IMAGE_MAX = 25 * 1024 * 1024;
const DEFAULT_VIDEO_MAX = 500 * 1024 * 1024;
const DEFAULT_USER_QUOTA = 10 * 1024 * 1024 * 1024;

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

    const data: MediaAssetData = {
      schemaVersion: MEDIA_ASSET_SCHEMA_VERSION,
      status: 'PENDING',
      storageKey: upload.key,
      filename: this.cleanFilename(dto.filename),
      mimeType: dto.mimeType,
      sizeBytes: dto.sizeBytes,
      capturedAt: dto.capturedAt ?? null,
      source,
      provider: source === 'apple-photos-picker' ? 'APPLE_PHOTOS' : source === 'device-picker' ? 'DEVICE' : null,
      favorite: false,
      albumIds,
      tags: normalizeMediaTags([
        ...this.sourceTags(source),
        ...(dto.tags ?? []).map((label) => ({ label, source: 'owner' as const })),
      ]),
      linkedObservationIds: dto.linkedObservationId ? [dto.linkedObservationId] : [],
      createdFrom: 'UPLOAD',
      uploadExpiresAt: new Date(Date.now() + upload.expiresIn * 1000).toISOString(),
      completedAt: null,
      sha256: null,
    };

    const asset = await this.prisma.telemetry.create({
      data: {
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ASSET_EVENT,
        userId,
        petId: dto.petId,
        data: data as Prisma.InputJsonValue,
      },
      select: { id: true, createdAt: true },
    });

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
    const entry = await this.requireAssetEntry(userId, dto.assetId);
    const data = this.parseAsset(entry.data);
    if (!data) throw new BadRequestException('Media asset metadata is invalid');
    if (data.status === 'READY') return this.decorateAsset(entry.id, entry.createdAt, data);
    if (data.status !== 'PENDING') throw new BadRequestException('Media upload is not pending');

    let object;
    try {
      object = await this.storage.headObject(data.storageKey);
    } catch {
      throw new BadRequestException('Uploaded media object could not be verified');
    }

    const sizeMatches = object.sizeBytes === data.sizeBytes;
    const contentTypeMatches = !object.contentType || object.contentType === data.mimeType;
    if (!sizeMatches || !contentTypeMatches) {
      await this.storage.deleteFile(data.storageKey).catch(() => undefined);
      const failed = { ...data, status: 'FAILED' as const, completedAt: new Date().toISOString() };
      await this.prisma.telemetry.update({
        where: { id: entry.id },
        data: { data: failed as Prisma.InputJsonValue },
      });
      throw new BadRequestException('Uploaded media did not match the declared size or content type');
    }

    if (dto.sha256 && !/^[a-f0-9]{64}$/i.test(dto.sha256)) {
      throw new BadRequestException('sha256 must be a 64-character hexadecimal digest');
    }

    const ready: MediaAssetData = {
      ...data,
      status: 'READY',
      uploadExpiresAt: null,
      completedAt: new Date().toISOString(),
      sha256: dto.sha256?.toLowerCase() ?? null,
    };
    await this.prisma.telemetry.update({
      where: { id: entry.id },
      data: { data: ready as Prisma.InputJsonValue },
    });
    return this.decorateAsset(entry.id, entry.createdAt, ready);
  }

  async library(userId: string, query: MediaLibraryQueryDto) {
    await this.requireOwnedPet(userId, query.petId);
    const limit = Math.max(1, Math.min(100, query.limit ?? 60));
    const entries = await this.prisma.telemetry.findMany({
      where: {
        userId,
        petId: query.petId,
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ASSET_EVENT,
      },
      orderBy: { createdAt: 'desc' },
      take: 300,
      select: { id: true, createdAt: true, data: true },
    });

    const ready = entries
      .map((entry) => {
        const data = this.parseAsset(entry.data);
        return data?.status === 'READY' ? { ...entry, parsed: data } : null;
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
      .filter((entry) => this.matchesQuery(entry.parsed, entry.createdAt, query))
      .slice(0, limit);

    const assets = await Promise.all(
      ready.map((entry) => this.decorateAsset(entry.id, entry.createdAt, entry.parsed)),
    );
    const albums = await this.albums(userId, query.petId, entries);
    const usedBytes = ready.reduce((sum, entry) => sum + entry.parsed.sizeBytes, 0);

    return {
      petId: query.petId,
      assets,
      albums,
      storage: {
        usedBytes,
        quotaBytes: this.userQuotaBytes,
        storageConfigured: this.storage.isConfigured(),
      },
      importCapabilities: {
        devicePicker: true,
        appleSystemPicker: true,
        googlePhotosPicker: true,
        googlePhotosBroadLibrarySync: false,
      },
    };
  }

  async albums(userId: string, petId: string, prefetchedAssetEntries?: Array<{ id: string; createdAt: Date; data: Prisma.JsonValue | null }>) {
    await this.requireOwnedPet(userId, petId);
    const [albumEntries, assetEntries] = await Promise.all([
      this.prisma.telemetry.findMany({
        where: { userId, petId, source: MEDIA_LIBRARY_SOURCE, event: MEDIA_ALBUM_EVENT },
        orderBy: { createdAt: 'asc' },
        select: { id: true, createdAt: true, data: true },
      }),
      prefetchedAssetEntries
        ? Promise.resolve(prefetchedAssetEntries)
        : this.prisma.telemetry.findMany({
            where: { userId, petId, source: MEDIA_LIBRARY_SOURCE, event: MEDIA_ASSET_EVENT },
            orderBy: { createdAt: 'desc' },
            take: 500,
            select: { id: true, createdAt: true, data: true },
          }),
    ]);

    const assets = assetEntries
      .map((entry) => {
        const data = this.parseAsset(entry.data);
        return data?.status === 'READY' ? { id: entry.id, createdAt: entry.createdAt, data } : null;
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null);

    const system = SYSTEM_ALBUMS.map((album) => ({
      id: album.id,
      name: album.name,
      description: album.description,
      icon: album.icon,
      kind: 'SMART' as const,
      count: assets.filter((asset) => album.matches(asset.data, asset.createdAt)).length,
    }));

    const custom = albumEntries
      .map((entry) => {
        const data = this.parseAlbum(entry.data);
        if (!data) return null;
        return {
          id: entry.id,
          name: data.name,
          description: data.description,
          icon: 'folder',
          kind: 'USER' as const,
          count: assets.filter((asset) => asset.data.albumIds.includes(entry.id)).length,
          coverAssetId: data.coverAssetId,
        };
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null);

    return [...system, ...custom];
  }

  async createAlbum(userId: string, dto: CreateMediaAlbumDto) {
    await this.requireOwnedPet(userId, dto.petId);
    const existing = await this.prisma.telemetry.count({
      where: { userId, petId: dto.petId, source: MEDIA_LIBRARY_SOURCE, event: MEDIA_ALBUM_EVENT },
    });
    if (existing >= 40) throw new BadRequestException('A pet can have at most 40 custom albums');

    const data: MediaAlbumData = {
      schemaVersion: MEDIA_ALBUM_SCHEMA_VERSION,
      name: dto.name.trim(),
      description: dto.description?.trim() || null,
      petId: dto.petId,
      kind: 'USER',
      coverAssetId: null,
    };
    if (!data.name) throw new BadRequestException('Album name is required');

    const album = await this.prisma.telemetry.create({
      data: {
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ALBUM_EVENT,
        userId,
        petId: dto.petId,
        data: data as Prisma.InputJsonValue,
      },
      select: { id: true, createdAt: true },
    });
    return { id: album.id, ...data, createdAt: album.createdAt.toISOString(), count: 0 };
  }

  async updateAsset(userId: string, assetId: string, dto: UpdateMediaAssetDto) {
    const entry = await this.requireAssetEntry(userId, assetId);
    const data = this.parseAsset(entry.data);
    if (!data) throw new BadRequestException('Media asset metadata is invalid');
    const petId = entry.petId;
    if (!petId) throw new BadRequestException('Media asset is missing pet context');

    const albumIds = dto.albumIds
      ? await this.validateAlbumIds(userId, petId, dto.albumIds)
      : data.albumIds;
    const systemTags = data.tags.filter((tag) => tag.source !== 'owner');
    const ownerTags = dto.tags
      ? dto.tags.map((label) => ({ label, source: 'owner' as const }))
      : data.tags.filter((tag) => tag.source === 'owner');
    const updated: MediaAssetData = {
      ...data,
      favorite: dto.favorite ?? data.favorite,
      albumIds,
      tags: normalizeMediaTags([...systemTags, ...ownerTags]),
    };

    await this.prisma.telemetry.update({
      where: { id: entry.id },
      data: { data: updated as Prisma.InputJsonValue },
    });
    return this.decorateAsset(entry.id, entry.createdAt, updated);
  }

  async deleteAsset(userId: string, assetId: string) {
    const entry = await this.requireAssetEntry(userId, assetId);
    const data = this.parseAsset(entry.data);
    if (!data) throw new BadRequestException('Media asset metadata is invalid');

    await this.storage.deleteFile(data.storageKey).catch(() => undefined);
    await this.prisma.telemetry.delete({ where: { id: entry.id } });
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
      pollingConfig: payload.pollingConfig ?? null,
      privacy:
        'Woof receives only the photos and videos you explicitly choose in Google Photos. The OAuth token is used for this request and is not written to the media timeline.',
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
      return { ready: false, imported: [], pollingConfig: session.pollingConfig ?? null };
    }

    const selected = await this.listGooglePickedItems(dto.accessToken, dto.sessionId);
    if (selected.length > 50) {
      throw new BadRequestException('Import at most 50 Google Photos items at a time');
    }

    const imported = [];
    for (const item of selected) {
      const mediaFile = this.asObject(item.mediaFile);
      const mimeType = typeof mediaFile.mimeType === 'string' ? mediaFile.mimeType : '';
      const baseUrl = typeof mediaFile.baseUrl === 'string' ? mediaFile.baseUrl : '';
      const filename =
        typeof mediaFile.filename === 'string' ? mediaFile.filename : `google-photo-${item.id ?? Date.now()}`;
      if (!ALLOWED_MIME_TYPES.has(mimeType) || !baseUrl) continue;

      const isVideo = mimeType.startsWith('video/');
      const downloadUrl = `${baseUrl}=${isVideo ? 'dv' : 'd'}`;
      const mediaResponse = await fetch(downloadUrl, {
        headers: { Authorization: `Bearer ${dto.accessToken}` },
      });
      if (!mediaResponse.ok) continue;
      const declared = Number(mediaResponse.headers.get('content-length') || 0);
      const maxBytes = isVideo ? this.videoMaxBytes : this.imageMaxBytes;
      if (declared > maxBytes) continue;
      const bytes = Buffer.from(await mediaResponse.arrayBuffer());
      this.assertAllowedMedia(mimeType, bytes.byteLength);
      await this.enforceQuota(userId, bytes.byteLength);

      const stored = await this.storage.uploadPrivateBytes({
        bytes,
        filename,
        contentType: mimeType,
        folder: `private/media/${userId}/${dto.petId}`,
      });
      const assetData: MediaAssetData = {
        schemaVersion: MEDIA_ASSET_SCHEMA_VERSION,
        status: 'READY',
        storageKey: stored.key,
        filename: this.cleanFilename(filename),
        mimeType,
        sizeBytes: bytes.byteLength,
        capturedAt: typeof item.createTime === 'string' ? item.createTime : null,
        source: 'google-photos-picker',
        provider: 'GOOGLE_PHOTOS',
        providerItemId: typeof item.id === 'string' ? item.id : null,
        favorite: false,
        albumIds,
        tags: normalizeMediaTags([
          { label: 'imported', source: 'system' },
          { label: 'google photos', source: 'system' },
        ]),
        linkedObservationIds: [],
        createdFrom: 'IMPORT',
        completedAt: new Date().toISOString(),
        sha256: null,
      };
      const created = await this.prisma.telemetry.create({
        data: {
          source: MEDIA_LIBRARY_SOURCE,
          event: MEDIA_ASSET_EVENT,
          userId,
          petId: dto.petId,
          data: assetData as Prisma.InputJsonValue,
        },
        select: { id: true, createdAt: true },
      });
      imported.push(await this.decorateAsset(created.id, created.createdAt, assetData));
    }

    await fetch(`${GOOGLE_PICKER_BASE}/sessions/${encodeURIComponent(dto.sessionId)}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${dto.accessToken}` },
    }).catch(() => undefined);

    await this.prisma.telemetry.create({
      data: {
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_IMPORT_EVENT,
        userId,
        petId: dto.petId,
        data: {
          provider: 'GOOGLE_PHOTOS',
          importedCount: imported.length,
          occurredAt: new Date().toISOString(),
        } as Prisma.InputJsonValue,
      },
    });

    return { ready: true, imported };
  }

  async exportToGooglePhotos(userId: string, dto: GooglePhotosExportDto) {
    await this.requireOwnedPet(userId, dto.petId);
    if (dto.assetIds.length === 0) throw new BadRequestException('Choose at least one media asset');

    const entries = await this.prisma.telemetry.findMany({
      where: {
        id: { in: dto.assetIds },
        userId,
        petId: dto.petId,
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ASSET_EVENT,
      },
      select: { id: true, data: true },
    });
    if (entries.length !== new Set(dto.assetIds).size) {
      throw new ForbiddenException('One or more media assets are unavailable');
    }

    const uploadTokens: Array<{ uploadToken: string; fileName: string; assetId: string }> = [];
    for (const entry of entries) {
      const data = this.parseAsset(entry.data);
      if (!data || data.status !== 'READY') continue;
      const bytes = await this.storage.getObjectBytes(data.storageKey, this.videoMaxBytes);
      const uploadResponse = await fetch(`${GOOGLE_LIBRARY_BASE}/uploads`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${dto.accessToken}`,
          'Content-Type': 'application/octet-stream',
          'X-Goog-Upload-Content-Type': data.mimeType,
          'X-Goog-Upload-Protocol': 'raw',
        },
        body: new Uint8Array(bytes),
      });
      if (!uploadResponse.ok) continue;
      const uploadToken = (await uploadResponse.text()).trim();
      if (uploadToken) uploadTokens.push({ uploadToken, fileName: data.filename, assetId: entry.id });
    }

    if (uploadTokens.length === 0) {
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

    await this.prisma.telemetry.create({
      data: {
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_EXPORT_EVENT,
        userId,
        petId: dto.petId,
        data: {
          provider: 'GOOGLE_PHOTOS',
          requestedCount: dto.assetIds.length,
          uploadedCount: uploadTokens.length,
          occurredAt: new Date().toISOString(),
        } as Prisma.InputJsonValue,
      },
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
    const whereIds = dto.assetIds?.length ? { id: { in: dto.assetIds } } : {};
    const entries = await this.prisma.telemetry.findMany({
      where: {
        ...whereIds,
        userId,
        petId: dto.petId,
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ASSET_EVENT,
      },
      orderBy: { createdAt: 'asc' },
      take: 100,
      select: { id: true, createdAt: true, data: true },
    });

    const assets = await Promise.all(
      entries.map(async (entry) => {
        const data = this.parseAsset(entry.data);
        if (!data || data.status !== 'READY') return null;
        return {
          id: entry.id,
          filename: data.filename,
          mimeType: data.mimeType,
          sizeBytes: data.sizeBytes,
          capturedAt: data.capturedAt,
          createdAt: entry.createdAt.toISOString(),
          source: data.source,
          favorite: data.favorite,
          tags: data.tags,
          linkedObservationIds: data.linkedObservationIds,
          downloadUrl: await this.storage.getSignedUrl(data.storageKey, 900),
          downloadUrlExpiresInSeconds: 900,
        };
      }),
    );

    return {
      schemaVersion: 'woof-media-export-v1',
      generatedAt: new Date().toISOString(),
      pet: { id: pet.id, name: pet.name, species: pet.species },
      assets: assets.filter((asset): asset is NonNullable<typeof asset> => asset !== null),
      portability:
        'Download URLs are intentionally short-lived. Asset metadata remains portable and does not expose Woof object-storage keys.',
    };
  }

  private matchesQuery(asset: MediaAssetData, createdAt: Date, query: MediaLibraryQueryDto) {
    if (query.tag) {
      const tag = query.tag.trim().toLowerCase();
      if (!asset.tags.some((entry) => entry.label.toLowerCase() === tag)) return false;
    }
    if (!query.albumId) return true;
    if (query.albumId.startsWith('smart:')) {
      const smart = SYSTEM_ALBUMS.find((album) => album.id === query.albumId);
      return smart ? smart.matches(asset, createdAt) : false;
    }
    return asset.albumIds.includes(query.albumId);
  }

  private async decorateAsset(id: string, createdAt: Date, data: MediaAssetData) {
    const signedUrl = data.status === 'READY' ? await this.storage.getSignedUrl(data.storageKey, 900).catch(() => null) : null;
    return {
      id,
      createdAt: createdAt.toISOString(),
      filename: data.filename,
      mimeType: data.mimeType,
      mediaType: data.mimeType.startsWith('video/') ? 'video' : 'image',
      sizeBytes: data.sizeBytes,
      capturedAt: data.capturedAt,
      source: data.source,
      provider: data.provider ?? null,
      favorite: data.favorite,
      albumIds: data.albumIds,
      smartAlbumIds: SYSTEM_ALBUMS.filter((album) => album.matches(data, createdAt)).map((album) => album.id),
      tags: data.tags,
      linkedObservationIds: data.linkedObservationIds,
      url: signedUrl,
      urlExpiresInSeconds: signedUrl ? 900 : null,
      status: data.status,
    };
  }

  private async validateAlbumIds(userId: string, petId: string, albumIds: string[]) {
    const unique = [...new Set(albumIds)].slice(0, 12);
    const custom = unique.filter((id) => !id.startsWith('smart:'));
    if (!custom.length) return [];
    const owned = await this.prisma.telemetry.findMany({
      where: {
        id: { in: custom },
        userId,
        petId,
        source: MEDIA_LIBRARY_SOURCE,
        event: MEDIA_ALBUM_EVENT,
      },
      select: { id: true },
    });
    if (owned.length !== custom.length) {
      throw new ForbiddenException('One or more albums are unavailable');
    }
    return owned.map((album) => album.id);
  }

  private async enforceQuota(userId: string, incomingBytes: number) {
    const entries = await this.prisma.telemetry.findMany({
      where: { userId, source: MEDIA_LIBRARY_SOURCE, event: MEDIA_ASSET_EVENT },
      select: { data: true },
      take: 5000,
    });
    const used = entries.reduce((sum, entry) => {
      const data = this.parseAsset(entry.data);
      return data && data.status !== 'FAILED' && data.status !== 'DELETED' ? sum + data.sizeBytes : sum;
    }, 0);
    if (used + incomingBytes > this.userQuotaBytes) {
      throw new BadRequestException('This upload would exceed the private media storage quota');
    }
  }

  private assertAllowedMedia(mimeType: string, sizeBytes: number) {
    if (!ALLOWED_MIME_TYPES.has(mimeType)) {
      throw new BadRequestException('Unsupported media type');
    }
    const max = mimeType.startsWith('video/') ? this.videoMaxBytes : this.imageMaxBytes;
    if (sizeBytes > max) {
      throw new BadRequestException(
        mimeType.startsWith('video/') ? 'Video exceeds the media-library size limit' : 'Image exceeds the media-library size limit',
      );
    }
  }

  private assertStorageReady() {
    if (!this.storage.isConfigured()) {
      throw new ServiceUnavailableException('Private media storage is not configured');
    }
  }

  private sourceTags(source: MediaSource) {
    if (source === 'behavior-vision') return [{ label: 'behavior', source: 'behavior' as const }];
    if (source === 'health-lens') return [{ label: 'health', source: 'health' as const }];
    if (source === 'coach') return [{ label: 'training', source: 'coach' as const }];
    return [];
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
      const payload = await this.requireGoogleJson(response, 'Could not list selected Google Photos media');
      const pageItems = Array.isArray(payload.mediaItems) ? payload.mediaItems : [];
      items.push(...pageItems.filter((item): item is Record<string, unknown> => this.isObject(item)));
      pageToken = typeof payload.nextPageToken === 'string' ? payload.nextPageToken : null;
    } while (pageToken && items.length < 100);
    return items.slice(0, 100);
  }

  private googleHeaders(accessToken: string) {
    return {
      Authorization: `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    };
  }

  private async requireGoogleJson(response: Response, message: string) {
    const text = await response.text();
    if (!response.ok) {
      throw new ServiceUnavailableException(`${message} (${response.status})`);
    }
    try {
      const parsed = text ? JSON.parse(text) : {};
      return this.isObject(parsed) ? parsed : {};
    } catch {
      throw new ServiceUnavailableException(`${message}: invalid provider response`);
    }
  }

  private async requireAssetEntry(userId: string, assetId: string) {
    const entry = await this.prisma.telemetry.findFirst({
      where: { id: assetId, userId, source: MEDIA_LIBRARY_SOURCE, event: MEDIA_ASSET_EVENT },
      select: { id: true, userId: true, petId: true, createdAt: true, data: true },
    });
    if (!entry) throw new NotFoundException('Media asset not found');
    return entry;
  }

  private async requireOwnedPet(userId: string, petId: string) {
    const pet = await this.prisma.pet.findFirst({
      where: { id: petId, ownerId: userId },
      select: { id: true, name: true, species: true },
    });
    if (!pet) throw new ForbiddenException('You do not have access to this pet');
    return pet;
  }

  private parseAsset(value: Prisma.JsonValue | null): MediaAssetData | null {
    const data = this.asObject(value);
    if (
      data.schemaVersion !== MEDIA_ASSET_SCHEMA_VERSION ||
      typeof data.storageKey !== 'string' ||
      typeof data.filename !== 'string' ||
      typeof data.mimeType !== 'string' ||
      typeof data.sizeBytes !== 'number' ||
      !Array.isArray(data.albumIds) ||
      !Array.isArray(data.tags) ||
      !Array.isArray(data.linkedObservationIds)
    ) {
      return null;
    }
    return data as unknown as MediaAssetData;
  }

  private parseAlbum(value: Prisma.JsonValue | null): MediaAlbumData | null {
    const data = this.asObject(value);
    if (
      data.schemaVersion !== MEDIA_ALBUM_SCHEMA_VERSION ||
      typeof data.name !== 'string' ||
      typeof data.petId !== 'string'
    ) {
      return null;
    }
    return data as unknown as MediaAlbumData;
  }

  private asObject(value: unknown): Record<string, any> {
    if (!value || Array.isArray(value) || typeof value !== 'object') return {};
    return value as Record<string, any>;
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
