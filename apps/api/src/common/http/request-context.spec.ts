import type { NextFunction, Request, Response } from 'express';
import {
  normalizeRequestId,
  requestContextMiddleware,
  requestPathname,
  requestRouteTemplate,
  type RequestWithContext,
} from './request-context';

describe('request context privacy contracts', () => {
  it('accepts only bounded single-line request ids', () => {
    expect(normalizeRequestId('req-12345678')).toBe('req-12345678');
    expect(normalizeRequestId('short')).toBeNull();
    expect(normalizeRequestId('req-1234\nforged')).toBeNull();
    expect(normalizeRequestId('x'.repeat(129))).toBeNull();
  });

  it('strips query strings from response-facing paths', () => {
    expect(
      requestPathname({
        originalUrl: '/api/v1/users/user-secret?token=do-not-log&email=private@example.com',
        url: '/ignored',
      } as Request)
    ).toBe('/api/v1/users/user-secret');
  });

  it('uses low-cardinality route templates for operational telemetry', () => {
    expect(
      requestRouteTemplate({
        baseUrl: '/api/v1',
        route: { path: '/users/:userId' },
      } as unknown as Request)
    ).toBe('/api/v1/users/:userId');
    expect(requestRouteTemplate({ baseUrl: '/api/v1' } as Request)).toBe('unmatched');
  });

  it('echoes a valid caller request id for cross-service correlation', () => {
    const request = {
      headers: { 'x-request-id': 'trace-12345678' },
    } as unknown as RequestWithContext;
    const response = { setHeader: jest.fn() } as unknown as Response;
    const next = jest.fn() as NextFunction;

    requestContextMiddleware(request, response, next);

    expect(request.requestId).toBe('trace-12345678');
    expect(response.setHeader).toHaveBeenCalledWith('X-Request-ID', 'trace-12345678');
    expect(next).toHaveBeenCalledTimes(1);
  });

  it('generates a server request id when the caller id is unsafe', () => {
    const request = {
      headers: { 'x-request-id': 'bad\nheader' },
    } as unknown as RequestWithContext;
    const response = { setHeader: jest.fn() } as unknown as Response;
    const next = jest.fn() as NextFunction;

    requestContextMiddleware(request, response, next);

    expect(request.requestId).toMatch(/^[0-9a-f-]{36}$/);
    expect(response.setHeader).toHaveBeenCalledWith('X-Request-ID', request.requestId);
    expect(next).toHaveBeenCalledTimes(1);
  });
});
