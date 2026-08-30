const GIT_SHA_PATTERN = /^[0-9a-f]{40}$/;

export const UNKNOWN_RELEASE = 'unknown' as const;

export function resolveWebReleaseIdentity(
  value = process.env.NEXT_PUBLIC_WOOF_RELEASE_SHA
): string {
  const normalized = value?.trim().toLowerCase() ?? '';
  return GIT_SHA_PATTERN.test(normalized) ? normalized : UNKNOWN_RELEASE;
}

export function resolveReplayPolicy(value = process.env.NEXT_PUBLIC_SENTRY_REPLAY_ENABLED) {
  const enabled = value === 'true';
  return {
    enabled,
    sessionSampleRate: enabled ? 0.01 : 0,
    errorSampleRate: enabled ? 0.1 : 0,
  } as const;
}

type PrivacySafeBrowserEvent = {
  request?: unknown;
  user?: unknown;
  extra?: unknown;
  breadcrumbs?: unknown;
};

export function scrubBrowserSentryEvent<T extends PrivacySafeBrowserEvent>(event: T) {
  delete event.request;
  delete event.user;
  delete event.extra;
  delete event.breadcrumbs;
  return event;
}
