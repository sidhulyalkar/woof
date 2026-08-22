import {
  DeleteObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
  PutObjectCommand,
  S3Client,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as crypto from 'crypto';
import { createReadStream, createWriteStream } from 'fs';
import { rm, stat } from 'fs/promises';
import * as path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import type { ReadableStream as NodeReadableStream } from 'stream/web';

export interface UploadResult {
  key: string;
  url: string;
  bucket: string;
}

export interface PrivateUploadResult {
  key: string;
  bucket: string;
}

export interface PrivateUploadIntent {
  key: string;
  uploadUrl: string;
  expiresIn: number;
  requiredHeaders: Record<string, string>;
}

export interface PrivateObjectInfo {
  key: string;
  sizeBytes: number;
  contentType: string | null;
  etag: string | null;
}

@Injectable()
export class StorageService {
  private readonly logger = new Logger(StorageService.name);
  private readonly s3Client: S3Client | null;
  private readonly bucket: string;
  private readonly region: string;
  private readonly publicUrl: string | null;
  private readonly configured: boolean;

  constructor(private configService: ConfigService) {
    this.region = this.configService.get<string>('AWS_REGION') || 'auto';
    this.bucket = this.configService.get<string>('S3_BUCKET') || 'woof-uploads';
    this.publicUrl = this.configService.get<string>('S3_PUBLIC_URL') || null;

    const endpoint = this.configService.get<string>('S3_ENDPOINT');
    const accessKeyId = this.configService.get<string>('S3_ACCESS_KEY_ID');
    const secretAccessKey = this.configService.get<string>('S3_SECRET_ACCESS_KEY');

    // Private object storage does not need a public CDN URL. This is important for
    // health/behavior/library media, which must never become public by default.
    this.configured = Boolean(accessKeyId && secretAccessKey);

    this.s3Client = this.configured
      ? new S3Client({
          region: this.region,
          endpoint: endpoint || undefined,
          credentials: {
            accessKeyId: accessKeyId!,
            secretAccessKey: secretAccessKey!,
          },
        })
      : null;

    if (!this.configured) {
      this.logger.warn('Object storage is not configured; durable media storage is disabled');
    }
  }

  isConfigured() {
    return this.configured;
  }

  isPublicDeliveryConfigured() {
    return this.configured && Boolean(this.publicUrl);
  }

  async uploadFile(file: Express.Multer.File, folder = 'uploads'): Promise<UploadResult> {
    if (!this.publicUrl) {
      throw new ServiceUnavailableException(
        'Public media delivery is not configured in this environment',
      );
    }

    const stored = await this.uploadPrivateFile(file, folder);
    return {
      ...stored,
      url: `${this.publicUrl.replace(/\/$/, '')}/${stored.key}`,
    };
  }

  async uploadPrivateFile(
    file: Express.Multer.File,
    folder = 'private/uploads',
  ): Promise<PrivateUploadResult> {
    return this.uploadPrivateBytes({
      bytes: file.buffer,
      filename: file.originalname,
      contentType: file.mimetype,
      folder,
    });
  }

  async uploadPrivateBytes(input: {
    bytes: Buffer | Uint8Array;
    filename: string;
    contentType: string;
    folder?: string;
  }): Promise<PrivateUploadResult> {
    const client = this.requireClient();
    const folder = input.folder ?? 'private/uploads';
    const key = this.generateKey(input.filename, folder);
    const bytes = Buffer.isBuffer(input.bytes) ? input.bytes : Buffer.from(input.bytes);

    try {
      await client.send(
        new PutObjectCommand({
          Bucket: this.bucket,
          Key: key,
          Body: bytes,
          ContentLength: bytes.byteLength,
          ContentType: input.contentType,
          Metadata: {
            originalName: input.filename.slice(0, 240),
            size: String(bytes.byteLength),
          },
        }),
      );
      this.logger.log(`Private file uploaded successfully: ${key}`);
      return { key, bucket: this.bucket };
    } catch (error) {
      this.logStorageError('Failed to upload private file', error);
      throw error;
    }
  }

  async uploadPrivateWebStream(input: {
    body: ReadableStream<Uint8Array>;
    contentLength: number;
    filename: string;
    contentType: string;
    folder?: string;
  }): Promise<PrivateUploadResult> {
    if (!Number.isFinite(input.contentLength) || input.contentLength <= 0) {
      throw new ServiceUnavailableException(
        'Streaming media import requires a positive content length',
      );
    }
    const client = this.requireClient();
    const key = this.generateKey(input.filename, input.folder ?? 'private/uploads');
    const body = Readable.fromWeb(input.body as unknown as NodeReadableStream<Uint8Array>);

    try {
      await client.send(
        new PutObjectCommand({
          Bucket: this.bucket,
          Key: key,
          Body: body,
          ContentLength: input.contentLength,
          ContentType: input.contentType,
          Metadata: {
            originalName: input.filename.slice(0, 240),
            size: String(input.contentLength),
          },
        }),
      );
      this.logger.log(`Private streamed file uploaded successfully: ${key}`);
      return { key, bucket: this.bucket };
    } catch (error) {
      body.destroy();
      this.logStorageError('Failed to stream private file', error);
      throw error;
    }
  }

  async createPrivateUploadIntent(input: {
    filename: string;
    folder: string;
    contentType: string;
    expectedSizeBytes: number;
    expiresIn?: number;
  }): Promise<PrivateUploadIntent> {
    const client = this.requireClient();
    const expiresIn = Math.max(60, Math.min(1800, input.expiresIn ?? 900));
    const key = this.generateKey(input.filename, input.folder);
    const requiredHeaders = {
      'Content-Type': input.contentType,
      'x-amz-meta-expected-size': String(input.expectedSizeBytes),
    };

    const uploadUrl = await getSignedUrl(
      client,
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        ContentType: input.contentType,
        Metadata: { expectedSize: String(input.expectedSizeBytes) },
      }),
      { expiresIn },
    );

    return { key, uploadUrl, expiresIn, requiredHeaders };
  }

  async uploadFiles(
    files: Express.Multer.File[],
    folder = 'uploads',
  ): Promise<UploadResult[]> {
    return Promise.all(files.map((file) => this.uploadFile(file, folder)));
  }

  async deleteFile(key: string): Promise<void> {
    const client = this.requireClient();

    try {
      await client.send(new DeleteObjectCommand({ Bucket: this.bucket, Key: key }));
      this.logger.log(`File deleted successfully: ${key}`);
    } catch (error) {
      this.logStorageError('Failed to delete file', error);
      throw error;
    }
  }

  async getSignedUrl(key: string, expiresIn = 900): Promise<string> {
    const client = this.requireClient();
    const boundedExpiry = Math.max(60, Math.min(3600, expiresIn));

    try {
      return await getSignedUrl(
        client,
        new GetObjectCommand({ Bucket: this.bucket, Key: key }),
        { expiresIn: boundedExpiry },
      );
    } catch (error) {
      this.logStorageError('Failed to generate signed URL', error);
      throw error;
    }
  }

  async headObject(key: string): Promise<PrivateObjectInfo> {
    const client = this.requireClient();
    const response = await client.send(new HeadObjectCommand({ Bucket: this.bucket, Key: key }));
    return {
      key,
      sizeBytes: Number(response.ContentLength ?? 0),
      contentType: response.ContentType ?? null,
      etag: response.ETag?.replace(/^"|"$/g, '') ?? null,
    };
  }

  async getObjectHeader(key: string, bytes = 64): Promise<Buffer> {
    const client = this.requireClient();
    const bounded = Math.max(16, Math.min(4096, Math.round(bytes)));
    const response = await client.send(
      new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Range: `bytes=0-${bounded - 1}`,
      }),
    );
    if (!response.Body) throw new ServiceUnavailableException('Media object returned no body');

    const body = response.Body as typeof response.Body & {
      transformToByteArray?: () => Promise<Uint8Array>;
    };
    if (body.transformToByteArray) {
      return Buffer.from(await body.transformToByteArray()).subarray(0, bounded);
    }

    const chunks: Buffer[] = [];
    let total = 0;
    for await (const chunk of response.Body as AsyncIterable<Uint8Array>) {
      chunks.push(Buffer.from(chunk));
      total += chunk.byteLength;
      if (total >= bounded) break;
    }
    return Buffer.concat(chunks).subarray(0, bounded);
  }

  async downloadObjectToFile(key: string, destination: string, maxBytes: number): Promise<number> {
    const client = this.requireClient();
    const response = await client.send(new GetObjectCommand({ Bucket: this.bucket, Key: key }));
    const declared = Number(response.ContentLength ?? 0);
    if (declared > maxBytes) {
      throw new ServiceUnavailableException('Media object exceeds processing limit');
    }
    if (!response.Body) throw new ServiceUnavailableException('Media object returned no body');

    try {
      await pipeline(
        Readable.from(response.Body as AsyncIterable<Uint8Array>),
        createWriteStream(destination, { flags: 'wx' }),
      );
      const info = await stat(destination);
      if (info.size <= 0 || info.size > maxBytes) {
        throw new ServiceUnavailableException('Downloaded media exceeded processing bounds');
      }
      return info.size;
    } catch (error) {
      await rm(destination, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  async uploadPrivateFilePath(input: {
    filePath: string;
    filename: string;
    contentType: string;
    folder?: string;
  }): Promise<PrivateUploadResult> {
    const client = this.requireClient();
    const info = await stat(input.filePath);
    if (!info.isFile() || info.size <= 0) {
      throw new ServiceUnavailableException('Derivative output was not a readable file');
    }
    const key = this.generateKey(input.filename, input.folder ?? 'private/derivatives');
    await client.send(
      new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: createReadStream(input.filePath),
        ContentLength: info.size,
        ContentType: input.contentType,
        Metadata: { generated: 'true', size: String(info.size) },
      }),
    );
    return { key, bucket: this.bucket };
  }

  async getObjectBytes(key: string, maxBytes = 600 * 1024 * 1024): Promise<Buffer> {
    const client = this.requireClient();
    const response = await client.send(new GetObjectCommand({ Bucket: this.bucket, Key: key }));
    const declared = Number(response.ContentLength ?? 0);
    if (declared > maxBytes) {
      throw new ServiceUnavailableException('Media object exceeds the export size limit');
    }
    if (!response.Body) throw new ServiceUnavailableException('Media object returned no body');

    const body = response.Body as typeof response.Body & {
      transformToByteArray?: () => Promise<Uint8Array>;
      [Symbol.asyncIterator]?: () => AsyncIterator<Uint8Array>;
    };

    if (body.transformToByteArray) {
      const bytes = await body.transformToByteArray();
      if (bytes.byteLength > maxBytes) {
        throw new ServiceUnavailableException('Media object exceeds the export size limit');
      }
      return Buffer.from(bytes);
    }

    const chunks: Buffer[] = [];
    let total = 0;
    if (!body[Symbol.asyncIterator]) {
      throw new ServiceUnavailableException('Media object stream is not readable');
    }
    for await (const chunk of body as AsyncIterable<Uint8Array>) {
      total += chunk.byteLength;
      if (total > maxBytes) {
        throw new ServiceUnavailableException('Media object exceeds the export size limit');
      }
      chunks.push(Buffer.from(chunk));
    }
    return Buffer.concat(chunks);
  }

  validateFileType(file: Express.Multer.File, allowedTypes: string[]): boolean {
    return allowedTypes.includes(file.mimetype);
  }

  validateFileSize(file: Express.Multer.File, maxSizeBytes: number): boolean {
    return file.size <= maxSizeBytes;
  }

  private requireClient(): S3Client {
    if (!this.s3Client || !this.configured) {
      throw new ServiceUnavailableException(
        'Media storage is not configured in this environment',
      );
    }
    return this.s3Client;
  }

  private generateKey(filename: string, folder: string): string {
    const safeFolder =
      folder.replace(/[^a-zA-Z0-9/_-]/g, '').replace(/^\/+|\/+$/g, '') || 'uploads';
    const ext = path.extname(filename).toLowerCase().replace(/[^.a-z0-9]/g, '');
    const hash = crypto.randomBytes(16).toString('hex');
    return `${safeFolder}/${Date.now()}-${hash}${ext}`;
  }

  private logStorageError(message: string, error: unknown) {
    const detail = error instanceof Error ? error.message : 'unknown error';
    const stack = error instanceof Error ? error.stack : undefined;
    this.logger.error(`${message}: ${detail}`, stack);
  }
}
