import * as Sentry from '@sentry/node';
import { nodeProfilingIntegration } from '@sentry/profiling-node';

type StatusCarrier = {
  status?: unknown;
};

type PrivacySafeSentryEvent = {
  request?: unknown;
  user?: unknown;
  extra?: unknown;
  breadcrumbs?: unknown;
  spans?: Array<{
    data?: unknown;
    description?: string;
    op?: string;
  }>;
};

export function scrubSentryEvent<T extends PrivacySafeSentryEvent>(event: T) {
  delete event.request;
  delete event.user;
  delete event.extra;
  delete event.breadcrumbs;
  return event;
}

export function scrubSentryTransaction<T extends PrivacySafeSentryEvent>(event: T) {
  scrubSentryEvent(event);
  for (const span of event.spans ?? []) {
    delete span.data;
    if (span.description) {
      span.description = span.op || 'operation';
    }
  }
  return event;
}

export function initSentry() {
  if (process.env.SENTRY_DSN) {
    Sentry.init({
      dsn: process.env.SENTRY_DSN,
      environment: process.env.NODE_ENV || 'development',
      sendDefaultPii: false,
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
        return scrubSentryEvent(event);
      },
      beforeSendTransaction(event) {
        return scrubSentryTransaction(event);
      },
    });
  }
}
