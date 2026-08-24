import * as Sentry from '@sentry/node';
import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import type { Response } from 'express';
import {
  requestPathname,
  requestRouteTemplate,
  type RequestWithContext,
} from '../http/request-context';

type HttpErrorBody = {
  message?: string | string[];
};

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger(AllExceptionsFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<RequestWithContext>();

    const status =
      exception instanceof HttpException ? exception.getStatus() : HttpStatus.INTERNAL_SERVER_ERROR;
    const message =
      exception instanceof HttpException ? exception.getResponse() : 'Internal server error';
    const requestId = request.requestId ?? 'unassigned';
    const route = requestRouteTemplate(request);
    const logContext = `request_id=${requestId} method=${request.method} route=${route} status=${status}`;

    if (status >= HttpStatus.INTERNAL_SERVER_ERROR) {
      this.logger.error(logContext, exception instanceof Error ? exception.stack : undefined);

      if (process.env.NODE_ENV === 'production') {
        Sentry.captureException(exception, {
          tags: {
            request_id: requestId,
            http_method: request.method,
            http_status: String(status),
          },
          contexts: {
            http: {
              method: request.method,
              route,
              status,
            },
          },
        });
      }
    } else if (status === HttpStatus.TOO_MANY_REQUESTS) {
      this.logger.warn(logContext);
    }

    const errorBody =
      message && typeof message === 'object' && !Array.isArray(message)
        ? (message as HttpErrorBody)
        : undefined;
    const responseMessage =
      typeof message === 'string'
        ? message
        : Array.isArray(errorBody?.message)
          ? errorBody.message.join(' ')
          : (errorBody?.message ?? 'Internal server error');

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: requestPathname(request),
      requestId,
      message: responseMessage,
      error: exception instanceof HttpException ? exception.name : 'InternalServerError',
    });
  }
}
