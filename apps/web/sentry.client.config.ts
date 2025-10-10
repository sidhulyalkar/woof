import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',

  // Performance Monitoring
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // Session Replay
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,

  integrations: [
    Sentry.replayIntegration({
      maskAllText: false,
      blockAllMedia: false,
    }),
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
      const status = (error as any).status;
      if (status >= 400 && status < 500) {
        return null;
      }
    }

    return event;
  },
});
