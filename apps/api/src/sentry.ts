import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

export function initSentry() {
  if (process.env.SENTRY_DSN) {
    Sentry.init({
      dsn: process.env.SENTRY_DSN,
      environment: process.env.NODE_ENV || 'development',
      integrations: [
        nodeProfilingIntegration(),
        Sentry.prismaIntegration(),
        Sentry.httpIntegration(),
      ],
      // Performance Monitoring
      tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
      // Profiling
      profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
      // Ignore expected errors
      ignoreErrors: [
        'UnauthorizedException',
        'NotFoundException',
        'BadRequestException',
      ],
      beforeSend(event, hint) {
        // Don't send 4xx errors to Sentry
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
  }
}
