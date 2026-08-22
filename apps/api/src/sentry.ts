import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

type StatusCarrier = {
  status?: unknown;
};

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
      tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
      profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
      ignoreErrors: [
        'UnauthorizedException',
        'NotFoundException',
        'BadRequestException',
      ],
      beforeSend(event, hint) {
        const error = hint.originalException;
        if (error && typeof error === 'object' && 'status' in error) {
          const status = (error as StatusCarrier).status;
          if (typeof status === 'number' && status >= 400 && status < 500) {
            return null;
          }
        }
        return event;
      },
    });
  }
}
