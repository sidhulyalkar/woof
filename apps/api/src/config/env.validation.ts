import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().positive().default(4000),
  API_PREFIX: z.string().default('api/v1'),
  API_DOCS_ENABLED: z.enum(['true', 'false']).default('false'),
  DATABASE_URL: z.string().min(1, 'DATABASE_URL is required'),
  SHADOW_DATABASE_URL: z.string().optional(),
  JWT_SECRET: z.string().min(16, 'JWT_SECRET must be at least 16 characters'),
  JWT_EXPIRES_IN: z.string().default('7d'),
  CORS_ORIGIN: z.string().default('http://localhost:3000'),
  SENTRY_DSN: z.string().optional(),
  S3_ENDPOINT: z.string().optional(),
  S3_BUCKET: z.string().optional(),
  S3_ACCESS_KEY_ID: z.string().optional(),
  S3_SECRET_ACCESS_KEY: z.string().optional(),
  S3_PUBLIC_URL: z.string().optional(),
  AWS_REGION: z.string().optional(),
  ENABLE_ADVENTURE_SYSTEM: z.enum(['true', 'false']).optional(),
  ENABLE_DOGOS_AUTOPILOT: z.enum(['true', 'false']).optional(),
  ENABLE_DOGOS_OUR_STORY: z.enum(['true', 'false']).optional(),
  ENABLE_DOGOS_CONNECTORS: z.enum(['true', 'false']).optional(),
  ENABLE_DOGOS_CONCIERGE: z.enum(['true', 'false']).optional(),
  CONNECTOR_CREDENTIALS_KEY: z.string().optional(),
  MEDIA_LIBRARY_IMAGE_MAX_BYTES: z.coerce
    .number()
    .int()
    .min(1024 * 1024)
    .max(100 * 1024 * 1024)
    .optional(),
  MEDIA_LIBRARY_VIDEO_MAX_BYTES: z.coerce
    .number()
    .int()
    .min(10 * 1024 * 1024)
    .max(1024 * 1024 * 1024)
    .optional(),
  MEDIA_LIBRARY_USER_QUOTA_BYTES: z.coerce
    .number()
    .int()
    .min(100 * 1024 * 1024)
    .max(1024 * 1024 * 1024 * 1024)
    .optional(),
  MEDIA_DERIVATIVES_ENABLED: z.enum(['true', 'false']).default('false'),
  MEDIA_FFMPEG_PATH: z.string().min(1).default('ffmpeg'),
  MEDIA_FFPROBE_PATH: z.string().min(1).default('ffprobe'),
  VAPID_PUBLIC_KEY: z.string().optional(),
  VAPID_PRIVATE_KEY: z.string().optional(),
  N8N_WEBHOOK_SECRET: z.string().optional(),
  OPENAI_API_KEY: z.string().min(20).optional(),
  OPENAI_HEALTH_MODEL: z.string().default('gpt-5.6-luna'),
  OPENAI_HEALTH_TIMEOUT_MS: z.coerce.number().int().min(3000).max(30000).default(12000),
  BEHAVIOR_VISION_SERVICE_URL: z.string().url().optional(),
  BEHAVIOR_VISION_SERVICE_TOKEN: z.string().min(16).optional(),
  BEHAVIOR_VISION_TIMEOUT_MS: z.coerce.number().int().min(5000).max(90000).default(45000),
});

function connectorKeyIsValid(encoded: string | undefined) {
  if (!encoded) return false;
  return Buffer.from(encoded, 'base64').length === 32;
}

export function validateEnvironment(config: Record<string, unknown>) {
  const parsed = envSchema.safeParse(config);

  if (!parsed.success) {
    const details = parsed.error.issues
      .map((issue) => `${issue.path.join('.') || 'environment'}: ${issue.message}`)
      .join('; ');
    throw new Error(`Invalid Woof API configuration: ${details}`);
  }

  const env = parsed.data;

  if (env.CONNECTOR_CREDENTIALS_KEY && !connectorKeyIsValid(env.CONNECTOR_CREDENTIALS_KEY)) {
    throw new Error('CONNECTOR_CREDENTIALS_KEY must decode to exactly 32 bytes');
  }

  if (env.NODE_ENV === 'production') {
    const knownDevelopmentSecret = /^(dev-|change-this|your-)/i;
    if (env.JWT_SECRET.length < 32 || knownDevelopmentSecret.test(env.JWT_SECRET)) {
      throw new Error(
        'JWT_SECRET must be a non-development secret of at least 32 characters in production'
      );
    }

    if (env.VAPID_PRIVATE_KEY && knownDevelopmentSecret.test(env.VAPID_PRIVATE_KEY)) {
      throw new Error('VAPID_PRIVATE_KEY must be replaced with a production key before startup');
    }

    if (env.BEHAVIOR_VISION_SERVICE_URL && !env.BEHAVIOR_VISION_SERVICE_TOKEN) {
      throw new Error(
        'BEHAVIOR_VISION_SERVICE_TOKEN is required when the behavior vision service is enabled in production'
      );
    }

    const hasAnyStorageCredential = Boolean(env.S3_ACCESS_KEY_ID || env.S3_SECRET_ACCESS_KEY);
    if (hasAnyStorageCredential && !(env.S3_ACCESS_KEY_ID && env.S3_SECRET_ACCESS_KEY)) {
      throw new Error('S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be configured together');
    }

    if (
      env.MEDIA_DERIVATIVES_ENABLED === 'true' &&
      !(env.S3_ACCESS_KEY_ID && env.S3_SECRET_ACCESS_KEY && env.S3_BUCKET)
    ) {
      throw new Error(
        'Private object storage must be configured when MEDIA_DERIVATIVES_ENABLED=true'
      );
    }

    if (
      env.ENABLE_DOGOS_CONNECTORS === 'true' &&
      !connectorKeyIsValid(env.CONNECTOR_CREDENTIALS_KEY)
    ) {
      throw new Error(
        'CONNECTOR_CREDENTIALS_KEY is required and must decode to 32 bytes when connectors are enabled in production'
      );
    }
  }

  return { ...config, ...env };
}
