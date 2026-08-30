import * as Sentry from '@sentry/nextjs';
import {
  resolveWebReleaseIdentity,
  scrubBrowserSentryEvent,
} from './src/lib/observability/sentry-policy';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',
  release: resolveWebReleaseIdentity(),

  // Performance Monitoring
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  integrations: [Sentry.httpIntegration()],

  // Ignore expected errors
  ignoreErrors: ['NEXT_NOT_FOUND', 'NEXT_REDIRECT'],

  beforeSend(event) {
    // Don't send in development
    if (process.env.NODE_ENV !== 'production') {
      return null;
    }

    return scrubBrowserSentryEvent(event);
  },
});
