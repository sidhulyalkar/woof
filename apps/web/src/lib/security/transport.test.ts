import { describe, expect, it } from 'vitest';
import { isLoopbackHostname, shouldRedirectToHttps } from './transport';

describe('Web transport policy', () => {
  it.each(['localhost', 'LOCALHOST', '127.0.0.1', '::1', '[::1]'])(
    'recognizes %s as loopback',
    (hostname) => {
      expect(isLoopbackHostname(hostname)).toBe(true);
    }
  );

  it.each(['woof.example.com', 'preview.woof.example.com', '10.0.0.5'])(
    'does not broaden loopback authority to %s',
    (hostname) => {
      expect(isLoopbackHostname(hostname)).toBe(false);
    }
  );

  it('requires HTTPS for a public production host', () => {
    expect(
      shouldRedirectToHttps({
        nodeEnv: 'production',
        forwardedProto: 'http',
        hostname: 'woof.example.com',
      })
    ).toBe(true);
  });

  it('accepts a public production request already proven HTTPS by the proxy', () => {
    expect(
      shouldRedirectToHttps({
        nodeEnv: 'production',
        forwardedProto: 'https',
        hostname: 'woof.example.com',
      })
    ).toBe(false);
  });

  it.each(['localhost', '127.0.0.1', '::1'])(
    'allows HTTP only for local production qualification on %s',
    (hostname) => {
      expect(
        shouldRedirectToHttps({
          nodeEnv: 'production',
          forwardedProto: null,
          hostname,
        })
      ).toBe(false);
    }
  );

  it('does not force HTTPS in non-production environments', () => {
    expect(
      shouldRedirectToHttps({
        nodeEnv: 'test',
        forwardedProto: null,
        hostname: 'woof.example.com',
      })
    ).toBe(false);
  });
});
