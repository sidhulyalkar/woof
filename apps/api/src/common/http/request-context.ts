import { randomUUID } from 'node:crypto';
import type { NextFunction, Request, Response } from 'express';

export const REQUEST_ID_HEADER = 'x-request-id';
const REQUEST_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/;

export type RequestWithContext = Request & {
  requestId?: string;
};

export function normalizeRequestId(value: unknown) {
  if (typeof value !== 'string') return null;
  return REQUEST_ID_PATTERN.test(value) ? value : null;
}

export function requestPathname(request: Request) {
  const rawUrl = request.originalUrl || request.url || '/';
  const queryIndex = rawUrl.indexOf('?');
  const pathname = queryIndex >= 0 ? rawUrl.slice(0, queryIndex) : rawUrl;
  return pathname || '/';
}

export function requestRouteTemplate(request: Request) {
  const route = request.route as { path?: unknown } | undefined;
  const routePath = typeof route?.path === 'string' ? route.path : null;
  if (!routePath) return 'unmatched';

  const baseUrl = typeof request.baseUrl === 'string' ? request.baseUrl : '';
  return `${baseUrl}${routePath}` || '/';
}

export function requestContextMiddleware(
  request: RequestWithContext,
  response: Response,
  next: NextFunction
) {
  const incoming = request.headers[REQUEST_ID_HEADER];
  const candidate = Array.isArray(incoming) ? incoming[0] : incoming;
  const requestId = normalizeRequestId(candidate) ?? randomUUID();

  request.requestId = requestId;
  response.setHeader('X-Request-ID', requestId);
  next();
}
