import * as Sentry from '@sentry/node';
import { ArgumentsHost, BadRequestException, Logger } from '@nestjs/common';
import type { Response } from 'express';
import { AllExceptionsFilter } from './all-exceptions.filter';
import type { RequestWithContext } from '../http/request-context';

jest.mock('@sentry/node', () => ({
  captureException: jest.fn(),
}));

describe('AllExceptionsFilter production privacy', () => {
  const originalNodeEnv = process.env.NODE_ENV;
  let errorSpy: jest.SpyInstance;
  let warnSpy: jest.SpyInstance;

  beforeEach(() => {
    process.env.NODE_ENV = 'production';
    jest.clearAllMocks();
    errorSpy = jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined);
    warnSpy = jest.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    errorSpy.mockRestore();
    warnSpy.mockRestore();
    process.env.NODE_ENV = originalNodeEnv;
  });

  function buildHost(exceptionRequest: Partial<RequestWithContext>) {
    const request = {
      method: 'GET',
      url: '/users/user-secret?token=secret-token',
      originalUrl: '/api/v1/users/user-secret?token=secret-token&email=private@example.com',
      baseUrl: '/api/v1',
      route: { path: '/users/:userId' },
      headers: { authorization: 'Bearer secret-token' },
      requestId: 'trace-12345678',
      user: { id: 'user-secret', email: 'private@example.com' },
      ...exceptionRequest,
    } as unknown as RequestWithContext;
    const response = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn().mockReturnThis(),
    } as unknown as Response;
    const host = {
      switchToHttp: () => ({
        getRequest: () => request,
        getResponse: () => response,
      }),
    } as ArgumentsHost;
    return { request, response, host };
  }

  it('captures only low-cardinality request context for server failures', () => {
    const filter = new AllExceptionsFilter();
    const { host, response } = buildHost({});
    const exception = new Error('database unavailable');

    filter.catch(exception, host);

    expect(Sentry.captureException).toHaveBeenCalledTimes(1);
    const captureOptions = (Sentry.captureException as jest.Mock).mock.calls[0]?.[1];
    expect(captureOptions).toEqual(
      expect.objectContaining({
        tags: {
          request_id: 'trace-12345678',
          http_method: 'GET',
          http_status: '500',
        },
        contexts: {
          http: {
            method: 'GET',
            route: '/api/v1/users/:userId',
            status: 500,
          },
        },
      })
    );
    const serialized = JSON.stringify(captureOptions);
    expect(serialized).not.toContain('secret-token');
    expect(serialized).not.toContain('private@example.com');
    expect(serialized).not.toContain('user-secret');
    expect(errorSpy).toHaveBeenCalledWith(
      'request_id=trace-12345678 method=GET route=/api/v1/users/:userId status=500',
      exception.stack
    );
    expect(response.json).toHaveBeenCalledWith(
      expect.objectContaining({
        statusCode: 500,
        path: '/api/v1/users/user-secret',
        requestId: 'trace-12345678',
      })
    );
  });

  it('does not send expected client failures to Sentry or error logs', () => {
    const filter = new AllExceptionsFilter();
    const { host, response } = buildHost({});

    filter.catch(new BadRequestException('invalid request'), host);

    expect(Sentry.captureException).not.toHaveBeenCalled();
    expect(errorSpy).not.toHaveBeenCalled();
    expect(warnSpy).not.toHaveBeenCalled();
    expect(response.json).toHaveBeenCalledWith(
      expect.objectContaining({
        statusCode: 400,
        path: '/api/v1/users/user-secret',
        requestId: 'trace-12345678',
        message: 'invalid request',
      })
    );
  });
});
