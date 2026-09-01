import { validateEnvironment } from './env.validation';

describe('validateEnvironment', () => {
  const base = {
    NODE_ENV: 'test',
    DATABASE_URL: 'postgresql://postgres:postgres@localhost:5432/woof_test',
    JWT_SECRET: 'test-secret-long-enough',
  };
  const productionBase = {
    ...base,
    NODE_ENV: 'production',
    JWT_SECRET: '8f3f89a744824fd99ee61797d67dc1023f6f7717762c4ab8',
    CORS_ORIGIN: 'https://app.example.com',
  };
  const credentialEncryptionKey = Buffer.alloc(32, 11).toString('base64');
  const behaviorReleasePin = {
    BEHAVIOR_VISION_RELEASE_ID: 'behavior-shadow-2026-08-27',
    BEHAVIOR_VISION_MODEL_VERSION: 'shadow-model-1',
    BEHAVIOR_VISION_FEATURE_VERSION: 'features-1',
    BEHAVIOR_VISION_ARTIFACT_SHA256:
      'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  };

  it('accepts the minimal test configuration', () => {
    const config = validateEnvironment(base);

    expect(config.NODE_ENV).toBe('test');
    expect(config.PORT).toBe(4000);
    expect(config.API_PREFIX).toBe('api/v1');
  });

  it('requires a database URL', () => {
    expect(() =>
      validateEnvironment({
        NODE_ENV: 'test',
        JWT_SECRET: base.JWT_SECRET,
      })
    ).toThrow(/DATABASE_URL/i);
  });

  it('rejects undersized JWT secrets', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        JWT_SECRET: 'too-short',
      })
    ).toThrow(/JWT_SECRET/i);
  });

  it('rejects short operational metrics credentials when configured', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        OPS_METRICS_TOKEN: 'short-token',
      })
    ).toThrow(/OPS_METRICS_TOKEN/i);
  });

  it('accepts a distinct long operational metrics credential', () => {
    const config = validateEnvironment({
      ...base,
      OPS_METRICS_TOKEN: 'ops-8f3f89a744824fd99ee61797d67dc1023f6f7717',
    });

    expect(config.OPS_METRICS_TOKEN).toBe('ops-8f3f89a744824fd99ee61797d67dc1023f6f7717');
  });

  it('requires a complete release pin when Behavior Vision service is configured', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        BEHAVIOR_VISION_SERVICE_URL: 'https://behavior.example.com',
        BEHAVIOR_VISION_RELEASE_ID: behaviorReleasePin.BEHAVIOR_VISION_RELEASE_ID,
      })
    ).toThrow(
      /release pin must include release ID, model version, feature version, and artifact SHA-256 together/i
    );
  });

  it('accepts a fully pinned Behavior Vision service in test environments', () => {
    const config = validateEnvironment({
      ...base,
      BEHAVIOR_VISION_SERVICE_URL: 'https://behavior.example.com',
      ...behaviorReleasePin,
    });

    expect(config.BEHAVIOR_VISION_ARTIFACT_SHA256).toBe(
      behaviorReleasePin.BEHAVIOR_VISION_ARTIFACT_SHA256
    );
  });

  it('rejects malformed Behavior Vision artifact hashes', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        ...behaviorReleasePin,
        BEHAVIOR_VISION_ARTIFACT_SHA256: 'not-a-sha256',
      })
    ).toThrow(/SHA-256/i);
  });

  it('requires stronger non-development secrets in production', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        JWT_SECRET: 'dev-this-is-long-but-still-not-production-safe',
      })
    ).toThrow(/production/i);
  });

  it('requires an authenticated model service in production', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        BEHAVIOR_VISION_SERVICE_URL: 'https://behavior.example.com',
        ...behaviorReleasePin,
      })
    ).toThrow(/BEHAVIOR_VISION_SERVICE_TOKEN/i);
  });

  it('requires Web Push VAPID keys to be configured as a pair in production', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        VAPID_PUBLIC_KEY: 'public-key-only',
      })
    ).toThrow(/VAPID_PUBLIC_KEY.*VAPID_PRIVATE_KEY.*together/i);

    expect(() =>
      validateEnvironment({
        ...productionBase,
        VAPID_PRIVATE_KEY: 'private-key-only',
      })
    ).toThrow(/VAPID_PUBLIC_KEY.*VAPID_PRIVATE_KEY.*together/i);
  });

  it('requires encrypted credential storage when Web Push is configured in production', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        VAPID_PUBLIC_KEY: 'public-production-key',
        VAPID_PRIVATE_KEY: 'private-production-key',
      })
    ).toThrow(/CONNECTOR_CREDENTIALS_KEY.*Web Push/i);
  });

  it('accepts a complete encrypted Web Push configuration in production', () => {
    const config = validateEnvironment({
      ...productionBase,
      VAPID_PUBLIC_KEY: 'public-production-key',
      VAPID_PRIVATE_KEY: 'private-production-key',
      CONNECTOR_CREDENTIALS_KEY: credentialEncryptionKey,
    });

    expect(config.VAPID_PUBLIC_KEY).toBe('public-production-key');
    expect(config.VAPID_PRIVATE_KEY).toBe('private-production-key');
    expect(config.CONNECTOR_CREDENTIALS_KEY).toBe(credentialEncryptionKey);
  });

  it('rejects malformed Web Push legacy plaintext compatibility cutoffs', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        PUSH_LEGACY_PLAINTEXT_READS_UNTIL: 'next-month',
      })
    ).toThrow(/PUSH_LEGACY_PLAINTEXT_READS_UNTIL.*ISO-8601/i);
  });

  it('rejects Web Push legacy plaintext compatibility windows beyond 30 production days', () => {
    const tooFar = new Date(Date.now() + 31 * 24 * 60 * 60 * 1000).toISOString();

    expect(() =>
      validateEnvironment({
        ...productionBase,
        PUSH_LEGACY_PLAINTEXT_READS_UNTIL: tooFar,
      })
    ).toThrow(/PUSH_LEGACY_PLAINTEXT_READS_UNTIL.*at most 30 days/i);
  });

  it('accepts a bounded Web Push legacy plaintext compatibility window in production', () => {
    const cutoff = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
    const config = validateEnvironment({
      ...productionBase,
      PUSH_LEGACY_PLAINTEXT_READS_UNTIL: cutoff,
    });

    expect(config.PUSH_LEGACY_PLAINTEXT_READS_UNTIL).toBe(cutoff);
  });

  it('accepts an already-expired Push legacy cutoff as an explicit disabled state', () => {
    const config = validateEnvironment({
      ...productionBase,
      PUSH_LEGACY_PLAINTEXT_READS_UNTIL: '2000-01-01T00:00:00Z',
    });

    expect(config.PUSH_LEGACY_PLAINTEXT_READS_UNTIL).toBe('2000-01-01T00:00:00Z');
  });

  it('fails closed when production CORS falls back to localhost', () => {
    expect(() =>
      validateEnvironment({
        ...productionBase,
        CORS_ORIGIN: undefined,
      })
    ).toThrow(/CORS_ORIGIN.*HTTPS origins/i);
  });

  it.each(['http://app.example.com', '*', 'null', 'https://app.example.com/path'])(
    'rejects unsafe production CORS origin %s',
    (corsOrigin) => {
      expect(() =>
        validateEnvironment({
          ...productionBase,
          CORS_ORIGIN: corsOrigin,
        })
      ).toThrow(/CORS_ORIGIN.*HTTPS origins/i);
    }
  );

  it('accepts multiple explicit HTTPS production browser origins', () => {
    const config = validateEnvironment({
      ...productionBase,
      CORS_ORIGIN: 'https://app.example.com, https://preview.example.com',
    });

    expect(config.CORS_ORIGIN).toBe('https://app.example.com, https://preview.example.com');
  });

  it('accepts a sufficiently strong production configuration', () => {
    const config = validateEnvironment(productionBase);

    expect(config.NODE_ENV).toBe('production');
    expect(config.CORS_ORIGIN).toBe('https://app.example.com');
  });
});
