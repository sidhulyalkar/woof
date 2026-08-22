import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';
import { MediaLibraryService } from './media-library.service';

const createdAt = new Date('2026-08-21T00:00:00.000Z');

function pendingAsset() {
  return {
    id: '11111111-1111-4111-8111-111111111111',
    ownerId: 'user-1',
    petId: 'pet-1',
    storageKey: 'private/media/user-1/pet-1/object.jpg',
    filename: 'nova.jpg',
    mimeType: 'image/jpeg',
    mediaType: 'image',
    sizeBytes: 1024n,
    capturedAt: null,
    source: 'device-picker',
    provider: 'DEVICE',
    providerItemId: null,
    favorite: false,
    status: 'PENDING',
    createdFrom: 'UPLOAD',
    sha256: null,
    width: null,
    height: null,
    durationMs: null,
    uploadExpiresAt: new Date(Date.now() + 60_000),
    completedAt: null,
    tags: [],
    linkedObservationIds: [],
    createdAt,
    updatedAt: createdAt,
    albumLinks: [],
  };
}

function makePrisma() {
  return {
    pet: {
      findFirst: jest.fn().mockResolvedValue({ id: 'pet-1', name: 'Nova', species: 'DOG' }),
    },
    mediaAsset: {
      create: jest.fn().mockResolvedValue(pendingAsset()),
      findFirst: jest.fn().mockResolvedValue(pendingAsset()),
      findMany: jest.fn().mockResolvedValue([]),
      update: jest
        .fn()
        .mockImplementation(({ data }: { data: Record<string, unknown> }) =>
          Promise.resolve({ ...pendingAsset(), ...data })
        ),
      delete: jest.fn().mockResolvedValue({}),
      aggregate: jest.fn().mockResolvedValue({ _sum: { sizeBytes: 0n } }),
    },
    mediaAlbum: {
      count: jest.fn().mockResolvedValue(0),
      findMany: jest.fn().mockResolvedValue([]),
      create: jest.fn(),
    },
    mediaAlbumAsset: {
      createMany: jest.fn().mockResolvedValue({ count: 0 }),
      deleteMany: jest.fn().mockResolvedValue({ count: 0 }),
    },
    mediaExternalReference: {
      findFirst: jest.fn().mockResolvedValue(null),
      create: jest.fn(),
    },
    telemetry: {
      create: jest.fn().mockResolvedValue({ id: 'event-1', createdAt }),
    },
    $transaction: jest
      .fn()
      .mockImplementation((operations: Array<Promise<unknown>>) => Promise.all(operations)),
  };
}

function makeStorage() {
  return {
    isConfigured: jest.fn().mockReturnValue(true),
    createPrivateUploadIntent: jest.fn().mockResolvedValue({
      key: 'private/media/user-1/pet-1/object.jpg',
      uploadUrl: 'https://private-storage.example/signed-put',
      expiresIn: 900,
      requiredHeaders: {
        'Content-Type': 'image/jpeg',
        'x-amz-meta-expected-size': '1024',
      },
    }),
    headObject: jest.fn().mockResolvedValue({
      key: 'private/media/user-1/pet-1/object.jpg',
      sizeBytes: 1024,
      contentType: 'image/jpeg',
      etag: 'etag',
    }),
    getSignedUrl: jest.fn().mockResolvedValue('https://private-storage.example/signed-get'),
    deleteFile: jest.fn().mockResolvedValue(undefined),
    uploadPrivateBytes: jest.fn(),
    uploadPrivateWebStream: jest.fn(),
    getObjectBytes: jest.fn(),
  };
}

function makeService(prisma = makePrisma(), storage = makeStorage()) {
  const config = { get: jest.fn().mockReturnValue(undefined) };
  return {
    prisma,
    storage,
    service: new MediaLibraryService(
      prisma as unknown as PrismaService,
      storage as unknown as StorageService,
      config as unknown as ConfigService
    ),
  };
}

describe('MediaLibraryService', () => {
  afterEach(() => jest.restoreAllMocks());

  it('creates a private direct-upload intent without persisting its signed URL', async () => {
    const { service, prisma, storage } = makeService();
    const result = await service.createUploadIntent('user-1', {
      petId: 'pet-1',
      filename: 'nova.jpg',
      mimeType: 'image/jpeg',
      sizeBytes: 1024,
      source: 'device-picker',
    });

    expect(result.assetId).toBe('11111111-1111-4111-8111-111111111111');
    expect(storage.createPrivateUploadIntent).toHaveBeenCalledWith(
      expect.objectContaining({ folder: 'private/media/user-1/pet-1' })
    );

    const createArgs = prisma.mediaAsset.create.mock.calls[0]?.[0];
    expect(createArgs).toBeDefined();
    expect(createArgs?.data).toEqual(
      expect.objectContaining({
        ownerId: 'user-1',
        petId: 'pet-1',
        storageKey: 'private/media/user-1/pet-1/object.jpg',
        sizeBytes: 1024n,
        status: 'PENDING',
      })
    );
    expect(createArgs?.data).not.toHaveProperty('uploadUrl');
    expect(createArgs?.data).not.toHaveProperty('signedUrl');
    expect(createArgs?.data).not.toHaveProperty('visibility', 'PUBLIC');
    expect(result.uploadUrl).toBe('https://private-storage.example/signed-put');
    expect(result.privacy.visibility).toBe('PRIVATE');
    expect(prisma.telemetry.create).not.toHaveBeenCalled();
  });

  it('rejects and deletes an upload whose stored size differs from the declared size', async () => {
    const prisma = makePrisma();
    const storage = makeStorage();
    storage.headObject.mockResolvedValue({
      key: 'private/media/user-1/pet-1/object.jpg',
      sizeBytes: 2048,
      contentType: 'image/jpeg',
      etag: 'etag',
    });
    const { service } = makeService(prisma, storage);

    await expect(
      service.completeUpload('user-1', {
        assetId: '11111111-1111-4111-8111-111111111111',
      })
    ).rejects.toThrow('did not match');
    expect(storage.deleteFile).toHaveBeenCalledWith('private/media/user-1/pet-1/object.jpg');
    expect(prisma.mediaAsset.update).toHaveBeenCalledWith(
      expect.objectContaining({ data: expect.objectContaining({ status: 'FAILED' }) })
    );
  });

  it('reserves pending bytes when enforcing the private storage quota', async () => {
    const prisma = makePrisma();
    prisma.mediaAsset.aggregate.mockResolvedValue({
      _sum: { sizeBytes: 10n * 1024n * 1024n * 1024n - 512n },
    });
    const { service } = makeService(prisma, makeStorage());

    await expect(
      service.createUploadIntent('user-1', {
        petId: 'pet-1',
        filename: 'too-much.jpg',
        mimeType: 'image/jpeg',
        sizeBytes: 1024,
      })
    ).rejects.toThrow('storage quota');
  });

  it('uses a Google Photos Picker token ephemerally and never writes it to telemetry', async () => {
    const { service, prisma } = makeService();
    const accessToken = 'ephemeral-google-oauth-token-never-persist-this';
    const fetchMock = jest.spyOn(global, 'fetch').mockResolvedValue({
      ok: true,
      status: 200,
      text: async () =>
        JSON.stringify({
          id: 'picker-session-1',
          pickerUri: 'https://photos.google.com/picker/example',
          pollingConfig: { pollInterval: '2s', timeoutIn: '60s' },
        }),
    } as Response);

    const result = await service.startGooglePhotosPicker('user-1', {
      petId: 'pet-1',
      accessToken,
      maxItemCount: 12,
    });

    expect(result.pickerUri).toContain('/autoclose');
    expect(fetchMock).toHaveBeenCalledWith(
      'https://photospicker.googleapis.com/v1/sessions',
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: `Bearer ${accessToken}` }),
      })
    );
    expect(JSON.stringify(prisma.telemetry.create.mock.calls)).not.toContain(accessToken);
    expect(JSON.stringify(prisma.mediaAsset.create.mock.calls)).not.toContain(accessToken);
  });
});
