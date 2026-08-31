import * as Sentry from '@sentry/nextjs';
import {
  resolveReplayPolicy,
  resolveWebRuntimeReleaseIdentity,
  scrubBrowserSentryEvent,
} from './src/lib/observability/sentry-policy';

const replay = resolveReplayPolicy();

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',
  release: resolveWebRuntimeReleaseIdentity(),

  // Performance Monitoring
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // Session Replay is privacy-closed by default and must be explicitly enabled at build time.
  replaysSessionSampleRate: replay.sessionSampleRate,
  replaysOnErrorSampleRate: replay.errorSampleRate,

  integrations: [
    ...(replay.enabled
      ? [
          Sentry.replayIntegration({
            maskAllText: true,
            blockAllMedia: true,
          }),
        ]
      : []),
    Sentry.browserTracingIntegration(),
  ],

  // Ignore expected errors
  ignoreErrors: [
    'ResizeObserver loop limit exceeded',
    'Non-Error promise rejection captured',
    'Network request failed',
  ],

  beforeSend(event, hint) {
    // Don't send in development
    if (process.env.NODE_ENV !== 'production') {
      return null;
    }

    // Filter out 4xx errors
    const error = hint.originalException;
    if (error && typeof error === 'object' && 'status' in error) {
      const status = (error as { status?: unknown }).status;
      if (typeof status === 'number' && status >= 400 && status < 500) {
        return null;
      }
    }

    return scrubBrowserSentryEvent(event);
  },
});
