import { isIP } from 'node:net';

type HeaderValue = string | string[] | undefined;

export type ThrottlerRequestLike = {
  headers?: Record<string, HeaderValue>;
  ip?: string;
  socket?: {
    remoteAddress?: string | null;
  };
};

function normalizeIp(value: string | null | undefined) {
  const candidate = value?.trim();
  return candidate && isIP(candidate) ? candidate : null;
}

export function isTrustedFlyRuntime(env: NodeJS.ProcessEnv = process.env) {
  return Boolean(env.FLY_APP_NAME?.trim() && env.FLY_MACHINE_ID?.trim());
}

export function flyClientIp(request: ThrottlerRequestLike) {
  const header = request.headers?.['fly-client-ip'];
  if (typeof header !== 'string') return null;
  return normalizeIp(header);
}

export async function clientIpTrackerForEnv(
  request: ThrottlerRequestLike,
  env: NodeJS.ProcessEnv
): Promise<string> {
  const directIp =
    normalizeIp(request.ip) ?? normalizeIp(request.socket?.remoteAddress) ?? 'unknown';

  if (!isTrustedFlyRuntime(env)) {
    return `direct:${directIp}`;
  }

  const proxyIp = flyClientIp(request);
  return proxyIp ? `fly:${proxyIp}` : `fly-fallback:${directIp}`;
}

export async function clientIpTracker(request: ThrottlerRequestLike): Promise<string> {
  return clientIpTrackerForEnv(request, process.env);
}
