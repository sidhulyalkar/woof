import { spawn } from 'node:child_process';
import { mkdtemp, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Interval } from '@nestjs/schedule';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';
import { declaredMimeMatches, sniffMediaFile } from './media-sniff';

export const MEDIA_DERIVATIVE_PROCESSOR_VERSION = 'media-derivatives-v1';
export type MediaDerivativeKind = 'THUMBNAIL' | 'POSTER' | 'PREVIEW';

const MAX_TEMP_BYTES = 600 * 1024 * 1024;

export function derivativeKindsForMediaType(mediaType: string): MediaDerivativeKind[] {
  return mediaType === 'video' ? ['POSTER', 'PREVIEW'] : mediaType === 'image' ? ['THUMBNAIL'] : [];
}

export function ffmpegArgs(kind: MediaDerivativeKind, input: string, output: string) {
  if (kind === 'THUMBNAIL') {
    return [
      '-y',
      '-i',
      input,
      '-vf',
      "scale='min(512,iw)':-2",
      '-frames:v',
      '1',
      '-map_metadata',
      '-1',
      '-q:v',
      '3',
      output,
    ];
  }
  if (kind === 'POSTER') {
    return [
      '-y',
      '-ss',
      '0.5',
      '-i',
      input,
      '-vf',
      "scale='min(720,iw)':-2",
      '-frames:v',
      '1',
      '-map_metadata',
      '-1',
      '-q:v',
      '3',
      output,
    ];
  }
  return [
    '-y',
    '-i',
    input,
    '-t',
    '6',
    '-an',
    '-vf',
    "scale='min(720,iw)':-2",
    '-map_metadata',
    '-1',
    '-c:v',
    'libx264',
    '-preset',
    'veryfast',
    '-crf',
    '28',
    '-pix_fmt',
    'yuv420p',
    '-movflags',
    '+faststart',
    output,
  ];
}

function derivativeFilename(kind: MediaDerivativeKind) {
  return kind === 'PREVIEW' ? 'preview.mp4' : kind === 'POSTER' ? 'poster.jpg' : 'thumbnail.jpg';
}

function derivativeMime(kind: MediaDerivativeKind) {
  return kind === 'PREVIEW' ? 'video/mp4' : 'image/jpeg';
}

function safeFailure(error: unknown) {
  return (error instanceof Error ? error.message : 'unknown media processing error').slice(0, 500);
}

type ProbeMetadata = {
  width: number | null;
  height: number | null;
  durationMs: number | null;
};

@Injectable()
export class MediaDerivativeWorkerService {
  private readonly logger = new Logger(MediaDerivativeWorkerService.name);
  private readonly enabled: boolean;
  private readonly ffmpegPath: string;
  private readonly ffprobePath: string;
  private working = false;

  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
    private readonly config: ConfigService,
  ) {
    this.enabled = String(this.config.get('MEDIA_DERIVATIVES_ENABLED') ?? 'false') === 'true';
    this.ffmpegPath = String(this.config.get('MEDIA_FFMPEG_PATH') ?? 'ffmpeg');
    this.ffprobePath = String(this.config.get('MEDIA_FFPROBE_PATH') ?? 'ffprobe');
  }

  @Interval('private-media-derivatives', 15_000)
  async tick() {
    if (!this.enabled || this.working || !this.storage.isConfigured()) return;
    this.working = true;
    try {
      await this.ensureDerivativeJobs();
      const job = await this.claimNextJob();
      if (job) await this.processJob(job);
    } catch (error) {
      this.logger.error(`Media derivative tick failed: ${safeFailure(error)}`);
    } finally {
      this.working = false;
    }
  }

  async processAssetNow(assetId: string) {
    if (!this.enabled) return { processed: false, reason: 'derivatives-disabled' };
    const asset = await this.prisma.mediaAsset.findUnique({ where: { id: assetId } });
    if (!asset || asset.status !== 'READY') return { processed: false, reason: 'asset-not-ready' };
    await this.ensureJobsForAsset(asset.id, asset.mediaType);
    let processed = 0;
    for (;;) {
      const job = await this.claimNextJob(asset.id);
      if (!job) break;
      await this.processJob(job);
      processed += 1;
    }
    return { processed: true, derivativeCount: processed };
  }

  private async ensureDerivativeJobs() {
    const assets = await this.prisma.mediaAsset.findMany({
      where: {
        status: 'READY',
        derivatives: { none: { processorVersion: MEDIA_DERIVATIVE_PROCESSOR_VERSION } },
      },
      take: 8,
      select: { id: true, mediaType: true },
    });
    for (const asset of assets) await this.ensureJobsForAsset(asset.id, asset.mediaType);
  }

  private async ensureJobsForAsset(assetId: string, mediaType: string) {
    const kinds = derivativeKindsForMediaType(mediaType);
    if (!kinds.length) return;
    await this.prisma.mediaDerivative.createMany({
      data: kinds.map((kind) => ({
        assetId,
        kind,
        processorVersion: MEDIA_DERIVATIVE_PROCESSOR_VERSION,
        storageKey: `pending/${assetId}/${MEDIA_DERIVATIVE_PROCESSOR_VERSION}/${kind.toLowerCase()}`,
        status: 'PENDING',
      })),
      skipDuplicates: true,
    });
  }

  private async claimNextJob(assetId?: string) {
    const candidate = await this.prisma.mediaDerivative.findFirst({
      where: {
        status: 'PENDING',
        processorVersion: MEDIA_DERIVATIVE_PROCESSOR_VERSION,
        ...(assetId ? { assetId } : {}),
        asset: { status: 'READY' },
      },
      orderBy: { createdAt: 'asc' },
      include: { asset: true },
    });
    if (!candidate) return null;
    const claimed = await this.prisma.mediaDerivative.updateMany({
      where: { id: candidate.id, status: 'PENDING' },
      data: { status: 'PROCESSING' },
    });
    return claimed.count === 1 ? candidate : null;
  }

  private async processJob(job: Awaited<ReturnType<MediaDerivativeWorkerService['claimNextJob']>>) {
    if (!job) return;
    const kind = job.kind as MediaDerivativeKind;
    const workspace = await mkdtemp(path.join(tmpdir(), 'woof-media-'));
    const inputPath = path.join(workspace, `original${path.extname(job.asset.filename) || '.bin'}`);
    const outputPath = path.join(workspace, derivativeFilename(kind));

    try {
      await this.storage.downloadObjectToFile(job.asset.storageKey, inputPath, MAX_TEMP_BYTES);
      const actualMime = await sniffMediaFile(inputPath);
      if (!declaredMimeMatches(actualMime, job.asset.mimeType)) {
        await this.prisma.$transaction([
          this.prisma.mediaAsset.update({
            where: { id: job.assetId },
            data: { status: 'QUARANTINED' },
          }),
          this.prisma.mediaDerivative.update({
            where: { id: job.id },
            data: {
              status: 'FAILED',
              metadata: {
                reason: 'mime-signature-mismatch',
                declaredMime: job.asset.mimeType,
                actualMime,
              } as Prisma.InputJsonValue,
            },
          }),
        ]);
        this.logger.warn(`Quarantined media asset ${job.assetId}: signature did not match MIME`);
        return;
      }

      const probe = await this.probe(inputPath);
      await this.prisma.mediaAsset.update({
        where: { id: job.assetId },
        data: {
          width: probe.width,
          height: probe.height,
          durationMs: probe.durationMs,
        },
      });

      await this.run(this.ffmpegPath, ffmpegArgs(kind, inputPath, outputPath), 120_000);
      const outputStat = await stat(outputPath);
      if (outputStat.size <= 0 || outputStat.size > 100 * 1024 * 1024) {
        throw new Error('Generated derivative had an invalid size');
      }
      const stored = await this.storage.uploadPrivateFilePath({
        filePath: outputPath,
        filename: derivativeFilename(kind),
        contentType: derivativeMime(kind),
        folder: `private/media-derivatives/${job.asset.ownerId}/${job.asset.petId}/${job.assetId}`,
      });

      await this.prisma.mediaDerivative.update({
        where: { id: job.id },
        data: {
          storageKey: stored.key,
          mimeType: derivativeMime(kind),
          sizeBytes: BigInt(outputStat.size),
          status: 'READY',
          metadata: {
            sourceMime: actualMime,
            width: probe.width,
            height: probe.height,
            durationMs: probe.durationMs,
            generatedAt: new Date().toISOString(),
            privacy: 'metadata-stripped',
            audioIncluded: false,
          } as Prisma.InputJsonValue,
        },
      });
    } catch (error) {
      await this.prisma.mediaDerivative.update({
        where: { id: job.id },
        data: {
          status: 'FAILED',
          metadata: {
            reason: safeFailure(error),
            failedAt: new Date().toISOString(),
          } as Prisma.InputJsonValue,
        },
      }).catch(() => undefined);
      this.logger.warn(`Derivative ${job.id} failed: ${safeFailure(error)}`);
    } finally {
      await rm(workspace, { recursive: true, force: true });
    }
  }

  private async probe(filePath: string): Promise<ProbeMetadata> {
    const stdout = await this.run(
      this.ffprobePath,
      ['-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', filePath],
      20_000,
    );
    const parsed = JSON.parse(stdout) as {
      streams?: Array<{ codec_type?: string; width?: number; height?: number; duration?: string }>;
      format?: { duration?: string };
    };
    const visual = parsed.streams?.find((stream) => stream.codec_type === 'video') ?? parsed.streams?.[0];
    const duration = Number(visual?.duration ?? parsed.format?.duration ?? NaN);
    return {
      width: Number.isFinite(visual?.width) ? Number(visual?.width) : null,
      height: Number.isFinite(visual?.height) ? Number(visual?.height) : null,
      durationMs: Number.isFinite(duration) ? Math.max(0, Math.round(duration * 1000)) : null,
    };
  }

  private run(command: string, args: string[], timeoutMs: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new Error(`${path.basename(command)} timed out`));
      }, timeoutMs);
      child.stdout.on('data', (chunk) => {
        if (stdout.length < 2_000_000) stdout += String(chunk);
      });
      child.stderr.on('data', (chunk) => {
        if (stderr.length < 100_000) stderr += String(chunk);
      });
      child.once('error', (error) => {
        clearTimeout(timer);
        reject(error);
      });
      child.once('close', (code) => {
        clearTimeout(timer);
        if (code === 0) resolve(stdout);
        else reject(new Error(`${path.basename(command)} exited ${code}: ${stderr.slice(-2000)}`));
      });
    });
  }
}
