import { describe, expect, it } from 'vitest';
import {
  UNKNOWN_RELEASE,
  resolveReplayPolicy,
  resolveWebReleaseIdentity,
  scrubBrowserSentryEvent,
} from './sentry-policy';

describe('Web telemetry policy', () => {
  it('trusts only exact Git SHA release identity', () => {
    expect(resolveWebReleaseIdentity('ABCDEF0123456789ABCDEF0123456789ABCDEF01')).toBe(
      'abcdef0123456789abcdef0123456789abcdef01'
    );
    expect(resolveWebReleaseIdentity('latest')).toBe(UNKNOWN_RELEASE);
    expect(resolveWebReleaseIdentity(undefined)).toBe(UNKNOWN_RELEASE);
  });

  it('keeps Session Replay disabled unless explicitly enabled', () => {
    expect(resolveReplayPolicy(undefined)).toEqual({
      enabled: false,
      sessionSampleRate: 0,
      errorSampleRate: 0,
    });
    expect(resolveReplayPolicy('false').enabled).toBe(false);
    expect(resolveReplayPolicy('true')).toEqual({
      enabled: true,
      sessionSampleRate: 0.01,
      errorSampleRate: 0.1,
    });
  });

  it('removes user/request/extra/breadcrumb fields before browser error transport', () => {
    const event = {
      message: 'example',
      request: { url: '/pets/private' },
      user: { id: 'user-1' },
      extra: { private: 'context' },
      breadcrumbs: [{ message: 'private action' }],
    };

    expect(scrubBrowserSentryEvent(event)).toEqual({ message: 'example' });
  });
});
