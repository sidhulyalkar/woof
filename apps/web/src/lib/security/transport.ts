export function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.replace(/^\[|\]$/g, '').toLowerCase();
  return normalized === 'localhost' || normalized === '127.0.0.1' || normalized === '::1';
}

export function shouldRedirectToHttps({
  nodeEnv,
  forwardedProto,
  hostname,
}: {
  nodeEnv: string | undefined;
  forwardedProto: string | null;
  hostname: string;
}): boolean {
  return nodeEnv === 'production' && forwardedProto !== 'https' && !isLoopbackHostname(hostname);
}
