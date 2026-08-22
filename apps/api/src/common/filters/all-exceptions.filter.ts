import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import type { Request, Response } from 'express';
import * as Sentry from '@sentry/node';

type RequestPrincipal = {
  sub?: string;
  id?: string;
  email?: string;
};

type RequestWithPrincipal = Request & {
  user?: RequestPrincipal;
};

type HttpErrorBody = {
  message?: string | string[];
};

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger(AllExceptionsFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<RequestWithPrincipal>();

    const status =
      exception instanceof HttpException
        ? exception.getStatus()
        : HttpStatus.INTERNAL_SERVER_ERROR;

    const message =
      exception instanceof HttpException
        ? exception.getResponse()
        : 'Internal server error';

    this.logger.error(
      `${request.method} ${request.url}`,
      exception instanceof Error ? exception.stack : exception,
    );

    if (process.env.NODE_ENV === 'production') {
      Sentry.captureException(exception, {
        contexts: {
          http: {
            method: request.method,
            url: request.url,
            headers: request.headers,
          },
        },
        user: request.user
          ? {
              id: request.user.sub ?? request.user.id,
              email: request.user.email,
            }
          : undefined,
      });
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
          : errorBody?.message ?? 'Internal server error';

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: request.url,
      message: responseMessage,
      error: exception instanceof HttpException ? exception.name : 'InternalServerError',
    });
  }
}
