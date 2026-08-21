import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';
import { MediaLibraryService } from './media-library.service';

function makePrisma() {
  return {
    pet: {
      findFirst: jest.fn().mockResolvedValue({ id: 'pet-1', name: 'Nova', species: 'DOG' }),
    },
    telemetry: {
      findMany: jest.fn().mockResolvedValue([]),
      findFirst: jest.fn(),
      count: jest.fn().mockResolvedValue(0),
      create: jest.fn().mockResolvedValue({
        id: '11111111-1111-4111-8111-111111111111',
        createdAt: new Date('2026-08-21T00:00:00.000Z'),
      }),
      update: jest.fn().mockResolvedValue({}),
      delete: jest.fn().mockResolvedValue({}),
    },
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
      config as unknown as ConfigService,
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
      expect.objectContaining({ folder: 'private/media/user-1/pet-1' }),
    );
    const persisted = JSON.stringify(prisma.telemetry.create.mock.calls[0][0]);
    expect(persisted).toContain('private/media/user-1/pet-1/object.jpg');
    expect(persisted).not.toContain('signed-put');
    expect(persisted).not.toContain('PUBLIC');
  });

  it('rejects and deletes an upload whose stored size differs from the declared size', async () => {
    const prisma = makePrisma();
    const storage = makeStorage();
    prisma.telemetry.findFirst.mockResolvedValue({
      id: '11111111-1111-4111-8111-111111111111',
      userId: 'user-1',
      petId: 'pet-1',
      createdAt: new Date('2026-08-21T00:00:00.000Z'),
      data: {
        schemaVersion: 'woof-media-asset-v1',
        status: 'PENDING',
        storageKey: 'private/media/user-1/pet-1/object.jpg',
        filename: 'nova.jpg',
        mimeType: 'image/jpeg',
        sizeBytes: 1024,
        capturedAt: null,
        source: 'device-picker',
        provider: 'DEVICE',
        favorite: false,
        albumIds: [],
        tags: [],
        linkedObservationIds: [],
        createdFrom: 'UPLOAD',
      },
    });
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
      }),
    ).rejects.toThrow('did not match');
    expect(storage.deleteFile).toHaveBeenCalled();
    expect(JSON.stringify(prisma.telemetry.update.mock.calls[0][0])).toContain('FAILED');
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
      }),
    );
    expect(JSON.stringify(prisma.telemetry.create.mock.calls)).not.toContain(accessToken);
  });
});
