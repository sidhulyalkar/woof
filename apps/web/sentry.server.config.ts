import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV || 'development',

  // Performance Monitoring
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  integrations: [Sentry.httpIntegration()],

  // Ignore expected errors
  ignoreErrors: ['NEXT_NOT_FOUND', 'NEXT_REDIRECT'],

  beforeSend(event, hint) {
    // Don't send in development
    if (process.env.NODE_ENV !== 'production') {
      return null;
    }

    return event;
  },
});
