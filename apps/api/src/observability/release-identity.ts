const GIT_SHA_PATTERN = /^[0-9a-f]{40}$/;

export const UNKNOWN_RELEASE = 'unknown' as const;

/**
 * Deployment identity is authoritative only when it is an exact Git commit SHA.
 * Arbitrary environment text must never become a trusted release identifier.
 */
export function resolveReleaseIdentity(value = process.env.WOOF_RELEASE_SHA): string {
  const normalized = value?.trim().toLowerCase() ?? '';
  return GIT_SHA_PATTERN.test(normalized) ? normalized : UNKNOWN_RELEASE;
}
