import { describe, expect, it } from 'vitest';
import { buildConnectSrc, getApiConnectOrigin } from './csp';

describe('CSP API connect source', () => {
  it('reduces a pathful API base to its origin so descendant endpoints remain allowed', () => {
    expect(getApiConnectOrigin('https://api.woof.test/api/v1')).toBe('https://api.woof.test');
    expect(buildConnectSrc('https://api.woof.test/api/v1')).toContain('https://api.woof.test');
    expect(buildConnectSrc('https://api.woof.test/api/v1')).not.toContain('/api/v1');
  });

  it('preserves a non-default development port in the allowed origin', () => {
    expect(getApiConnectOrigin('http://127.0.0.1:59999/api/v1')).toBe('http://127.0.0.1:59999');
  });

  it('does not broaden the policy when the configured API URL is missing or malformed', () => {
    expect(getApiConnectOrigin()).toBeNull();
    expect(getApiConnectOrigin('not a url')).toBeNull();
    expect(buildConnectSrc('not a url')).toBe("'self' https://vitals.vercel-insights.com");
  });
});
