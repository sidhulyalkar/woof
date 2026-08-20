import { validateEnvironment } from './env.validation';

describe('validateEnvironment', () => {
  const base = {
    NODE_ENV: 'test',
    DATABASE_URL: 'postgresql://postgres:postgres@localhost:5432/woof_test',
    JWT_SECRET: 'test-secret-long-enough',
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
      }),
    ).toThrow(/DATABASE_URL/i);
  });

  it('rejects undersized JWT secrets', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        JWT_SECRET: 'too-short',
      }),
    ).toThrow(/JWT_SECRET/i);
  });

  it('requires stronger non-development secrets in production', () => {
    expect(() =>
      validateEnvironment({
        ...base,
        NODE_ENV: 'production',
        JWT_SECRET: 'dev-this-is-long-but-still-not-production-safe',
      }),
    ).toThrow(/production/i);
  });

  it('accepts a sufficiently strong production secret', () => {
    const config = validateEnvironment({
      ...base,
      NODE_ENV: 'production',
      JWT_SECRET: '8f3f89a744824fd99ee61797d67dc1023f6f7717762c4ab8',
    });

    expect(config.NODE_ENV).toBe('production');
  });
});
