import { Injectable, Logger } from '@nestjs/common';
import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  GetObjectCommand,
} from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
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
  private readonly s3Client: S3Client;
  private readonly bucket: string;
  private readonly region: string;
  private readonly publicUrl: string;

  constructor(private configService: ConfigService) {
    this.region = this.configService.get('AWS_REGION') || 'auto';
    this.bucket = this.configService.get('S3_BUCKET') || 'woof-uploads';
    this.publicUrl =
      this.configService.get('S3_PUBLIC_URL') || 'https://uploads.woof.app';

    // Support both AWS S3 and Cloudflare R2
    this.s3Client = new S3Client({
      region: this.region,
      endpoint: this.configService.get('S3_ENDPOINT'), // For R2: https://[account-id].r2.cloudflarestorage.com
      credentials: {
        accessKeyId: this.configService.get('S3_ACCESS_KEY_ID') || '',
        secretAccessKey: this.configService.get('S3_SECRET_ACCESS_KEY') || '',
      },
    });
  }

  /**
   * Upload a file to S3/R2
   */
  async uploadFile(
    file: Express.Multer.File,
    folder: string = 'uploads',
  ): Promise<UploadResult> {
    try {
      const key = this.generateKey(file.originalname, folder);

      const command = new PutObjectCommand({
        Bucket: this.bucket,
        Key: key,
        Body: file.buffer,
        ContentType: file.mimetype,
        Metadata: {
          originalName: file.originalname,
          size: file.size.toString(),
        },
      });

      await this.s3Client.send(command);

      const url = `${this.publicUrl}/${key}`;

      this.logger.log(`File uploaded successfully: ${key}`);

      return {
        key,
        url,
        bucket: this.bucket,
      };
    } catch (error) {
      this.logger.error(`Failed to upload file: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Upload multiple files
   */
  async uploadFiles(
    files: Express.Multer.File[],
    folder: string = 'uploads',
  ): Promise<UploadResult[]> {
    return Promise.all(files.map((file) => this.uploadFile(file, folder)));
  }

  /**
   * Delete a file from S3/R2
   */
  async deleteFile(key: string): Promise<void> {
    try {
      const command = new DeleteObjectCommand({
        Bucket: this.bucket,
        Key: key,
      });

      await this.s3Client.send(command);

      this.logger.log(`File deleted successfully: ${key}`);
    } catch (error) {
      this.logger.error(`Failed to delete file: ${error.message}`, error.stack);
      throw error;
    }
  }

  /**
   * Get a signed URL for private files
   */
  async getSignedUrl(key: string, expiresIn: number = 3600): Promise<string> {
    try {
      const command = new GetObjectCommand({
        Bucket: this.bucket,
        Key: key,
      });

      return await getSignedUrl(this.s3Client, command, { expiresIn });
    } catch (error) {
      this.logger.error(
        `Failed to generate signed URL: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  /**
   * Generate a unique key for the file
   */
  private generateKey(filename: string, folder: string): string {
    const ext = path.extname(filename);
    const hash = crypto.randomBytes(16).toString('hex');
    const timestamp = Date.now();
    return `${folder}/${timestamp}-${hash}${ext}`;
  }

  /**
   * Validate file type
   */
  validateFileType(file: Express.Multer.File, allowedTypes: string[]): boolean {
    return allowedTypes.includes(file.mimetype);
  }

  /**
   * Validate file size
   */
  validateFileSize(file: Express.Multer.File, maxSizeBytes: number): boolean {
    return file.size <= maxSizeBytes;
  }
}
