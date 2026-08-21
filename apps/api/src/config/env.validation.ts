import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  PORT: z.coerce.number().int().positive().default(4000),
  API_PREFIX: z.string().default('api/v1'),
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
  VAPID_PUBLIC_KEY: z.string().optional(),
  VAPID_PRIVATE_KEY: z.string().optional(),
  N8N_WEBHOOK_SECRET: z.string().optional(),
  OPENAI_API_KEY: z.string().min(20).optional(),
  OPENAI_HEALTH_MODEL: z.string().default('gpt-5.6-luna'),
  OPENAI_HEALTH_TIMEOUT_MS: z.coerce.number().int().min(3000).max(30000).default(12000),
});

export function validateEnvironment(config: Record<string, unknown>) {
  const parsed = envSchema.safeParse(config);

  if (!parsed.success) {
    const details = parsed.error.issues
      .map((issue) => `${issue.path.join('.') || 'environment'}: ${issue.message}`)
      .join('; ');
    throw new Error(`Invalid Woof API configuration: ${details}`);
  }

  const env = parsed.data;

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
  }

  return { ...config, ...env };
}
