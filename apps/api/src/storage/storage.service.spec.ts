import { Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { StorageService } from './storage.service';

function service(overrides: Record<string, string | undefined> = {}) {
  const values: Record<string, string | undefined> = {
    AWS_REGION: 'us-west-2',
    S3_BUCKET: 'private-test-bucket',
    S3_ACCESS_KEY_ID: 'test-access-key',
    S3_SECRET_ACCESS_KEY: 'test-secret-key',
    ...overrides,
  };
  const config = {
    get: jest.fn((key: string) => values[key]),
  };
  return new StorageService(config as unknown as ConfigService);
}

function replaceClient(storage: StorageService, send: jest.Mock) {
  (storage as unknown as { s3Client: { send: jest.Mock } | null }).s3Client = { send };
}

function loggerSpies() {
  return {
    log: jest.spyOn(Logger.prototype, 'log').mockImplementation(() => undefined),
    warn: jest.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined),
    error: jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined),
  };
}

function serializedLogCalls(spies: ReturnType<typeof loggerSpies>) {
  return JSON.stringify([
    ...spies.log.mock.calls,
    ...spies.warn.mock.calls,
    ...spies.error.mock.calls,
  ]);
}

describe('StorageService private provider boundary', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('keeps generated private keys and filenames out of successful telemetry', async () => {
    const spies = loggerSpies();
    const storage = service();
    const send = jest.fn().mockResolvedValue({});
    replaceClient(storage, send);

    const result = await storage.uploadPrivateBytes({
      bytes: Buffer.from('private-bytes'),
      filename: 'owner-private-health-photo.jpg',
      contentType: 'image/jpeg',
      folder: 'private/health',
    });

    expect(result.bucket).toBe('private-test-bucket');
    expect(result.key).toMatch(/^private\/health\//);
    expect(send).toHaveBeenCalledTimes(1);
    expect(serializedLogCalls(spies)).not.toContain(result.key);
    expect(serializedLogCalls(spies)).not.toContain('owner-private-health-photo.jpg');
  });

  it('normalizes upload SDK failures without logging or rethrowing provider details', async () => {
    const privateMarker = 'PRIVATE_S3_PROVIDER_DETAIL owner-object-key token=secret';
    const spies = loggerSpies();
    const storage = service();
    const send = jest.fn().mockRejectedValue(new Error(privateMarker));
    replaceClient(storage, send);

    await expect(
      storage.uploadPrivateBytes({
        bytes: Buffer.from('private-bytes'),
        filename: 'owner-private-photo.jpg',
        contentType: 'image/jpeg',
      })
    ).rejects.toEqual(
      new ServiceUnavailableException('Media storage operation is temporarily unavailable')
    );

    expect(spies.error).toHaveBeenCalledWith(
      'Object storage provider failure operation=upload_private_bytes'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
    expect(serializedLogCalls(spies)).not.toContain('owner-private-photo.jpg');
  });

  it('normalizes delete SDK failures without logging the private object key', async () => {
    const privateMarker = 'PRIVATE_DELETE_PROVIDER_RESPONSE';
    const objectKey = 'private/health/user-specific-object.jpg';
    const spies = loggerSpies();
    const storage = service();
    const send = jest.fn().mockRejectedValue(new Error(privateMarker));
    replaceClient(storage, send);

    await expect(storage.deleteFile(objectKey)).rejects.toThrow(
      'Media storage operation is temporarily unavailable'
    );

    expect(spies.error).toHaveBeenCalledWith(
      'Object storage provider failure operation=delete_object'
    );
    expect(serializedLogCalls(spies)).not.toContain(privateMarker);
    expect(serializedLogCalls(spies)).not.toContain(objectKey);
  });

  it('preserves bounded application errors without misclassifying them as provider diagnostics', async () => {
    const spies = loggerSpies();
    const storage = service();
    const send = jest.fn().mockResolvedValue({ ContentLength: 2048, Body: {} });
    replaceClient(storage, send);

    await expect(storage.getObjectBytes('private/object.bin', 1024)).rejects.toThrow(
      'Media object exceeds the export size limit'
    );
    expect(spies.error).not.toHaveBeenCalled();
  });

  it('fails closed before provider access when private storage is unconfigured', async () => {
    const spies = loggerSpies();
    const storage = service({
      S3_ACCESS_KEY_ID: undefined,
      S3_SECRET_ACCESS_KEY: undefined,
    });

    await expect(
      storage.uploadPrivateBytes({
        bytes: Buffer.from('private-bytes'),
        filename: 'private.jpg',
        contentType: 'image/jpeg',
      })
    ).rejects.toThrow(/not configured/i);

    expect(spies.warn).toHaveBeenCalledWith(
      'Object storage is not configured; durable media storage is disabled'
    );
  });
});
