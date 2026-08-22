import {
  ArgumentsHost,
  Catch,
  ExceptionFilter,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import * as Sentry from '@sentry/node';
import { Request, Response } from 'express';

const SENSITIVE_HEADERS = new Set([
  'authorization',
  'cookie',
  'proxy-authorization',
  'x-api-key',
  'x-auth-token',
]);

type RequestWithOptionalUser = Request & {
  user?: {
    sub?: string;
    id?: string;
  };
};

export function redactRequestHeaders(headers: Request['headers']) {
  return Object.fromEntries(
    Object.entries(headers).filter(([name]) => !SENSITIVE_HEADERS.has(name.toLowerCase()))
  );
}

function normalizeHttpMessage(response: string | object) {
  if (typeof response === 'string') return response;
  if ('message' in response) return response.message;
  return 'Request failed';
}

@Catch()
export class AllExceptionsFilter implements ExceptionFilter {
  private readonly logger = new Logger(AllExceptionsFilter.name);

  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest<RequestWithOptionalUser>();

    const status =
      exception instanceof HttpException ? exception.getStatus() : HttpStatus.INTERNAL_SERVER_ERROR;
    const exceptionResponse =
      exception instanceof HttpException ? exception.getResponse() : 'Internal server error';
    const message = normalizeHttpMessage(exceptionResponse);
    const safePath = request.path || request.url.split('?')[0];

    this.logger.error(
      `${request.method} ${safePath}`,
      exception instanceof Error ? exception.stack : exception
    );

    if (process.env.NODE_ENV === 'production') {
      const userId = request.user?.sub ?? request.user?.id;
      Sentry.captureException(exception, {
        contexts: {
          http: {
            method: request.method,
            url: safePath,
            headers: redactRequestHeaders(request.headers),
          },
        },
        user: userId ? { id: userId } : undefined,
      });
    }

    response.status(status).json({
      statusCode: status,
      timestamp: new Date().toISOString(),
      path: safePath,
      message,
      error: exception instanceof HttpException ? exception.name : 'InternalServerError',
    });
  }
}
