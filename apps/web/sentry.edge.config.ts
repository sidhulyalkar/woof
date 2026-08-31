import * as Sentry from '@sentry/nextjs';
import {
  resolveWebRuntimeReleaseIdentity,
  scrubBrowserSentryEvent,
} from './src/lib/observability/sentry-policy';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',
  release: resolveWebRuntimeReleaseIdentity(),
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
  beforeSend(event) {
    if (process.env.NODE_ENV !== 'production') {
      return null;
    }
    return scrubBrowserSentryEvent(event);
  },
});
