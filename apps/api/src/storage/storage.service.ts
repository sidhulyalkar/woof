import { DeleteObjectCommand, GetObjectCommand, PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Injectable, Logger, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as crypto from 'crypto';
import * as path from 'path';

export interface UploadResult {
  key: string;
  url: string;
  bucket: string;
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

    this.configured = Boolean(accessKeyId && secretAccessKey && this.publicUrl);

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
      this.logger.warn('Object storage is not configured; media operations are disabled');
    }
  }

  async uploadFile(
    file: Express.Multer.File,
    folder = 'uploads',
  ): Promise<UploadResult> {
    const client = this.requireClient();
    const key = this.generateKey(file.originalname, folder);

    try {
      await client.send(
        new PutObjectCommand({
          Bucket: this.bucket,
          Key: key,
          Body: file.buffer,
          ContentType: file.mimetype,
          Metadata: {
            originalName: file.originalname,
            size: file.size.toString(),
          },
        }),
      );

      const url = `${this.publicUrl}/${key}`;
      this.logger.log(`File uploaded successfully: ${key}`);
      return { key, url, bucket: this.bucket };
    } catch (error: any) {
      this.logger.error(`Failed to upload file: ${error?.message || 'unknown error'}`, error?.stack);
      throw error;
    }
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
      await client.send(
        new DeleteObjectCommand({
          Bucket: this.bucket,
          Key: key,
        }),
      );
      this.logger.log(`File deleted successfully: ${key}`);
    } catch (error: any) {
      this.logger.error(`Failed to delete file: ${error?.message || 'unknown error'}`, error?.stack);
      throw error;
    }
  }

  async getSignedUrl(key: string, expiresIn = 3600): Promise<string> {
    const client = this.requireClient();

    try {
      return await getSignedUrl(
        client,
        new GetObjectCommand({ Bucket: this.bucket, Key: key }),
        { expiresIn },
      );
    } catch (error: any) {
      this.logger.error(
        `Failed to generate signed URL: ${error?.message || 'unknown error'}`,
        error?.stack,
      );
      throw error;
    }
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
    const safeFolder = folder.replace(/[^a-zA-Z0-9/_-]/g, '').replace(/^\/+|\/+$/g, '') || 'uploads';
    const ext = path.extname(filename).toLowerCase().replace(/[^.a-z0-9]/g, '');
    const hash = crypto.randomBytes(16).toString('hex');
    return `${safeFolder}/${Date.now()}-${hash}${ext}`;
  }
}
