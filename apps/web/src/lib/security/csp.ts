export function getApiConnectOrigin(apiUrl?: string): string | null {
  if (!apiUrl) return null;

  try {
    return new URL(apiUrl).origin;
  } catch {
    return null;
  }
}

export function buildConnectSrc(apiUrl?: string): string {
  const apiOrigin = getApiConnectOrigin(apiUrl);
  return ["'self'", apiOrigin, 'https://vitals.vercel-insights.com'].filter(Boolean).join(' ');
}
