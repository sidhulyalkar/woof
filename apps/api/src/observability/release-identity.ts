const GIT_SHA_PATTERN = /^[0-9a-f]{40}$/;

export const UNKNOWN_RELEASE = 'unknown' as const;

/**
 * Validate one candidate deployment identity. This function is deliberately
 * pure: ambient process state must never replace an explicitly supplied value.
 */
export function resolveReleaseIdentity(value: string | undefined): string {
  const normalized = value?.trim().toLowerCase() ?? '';
  return GIT_SHA_PATTERN.test(normalized) ? normalized : UNKNOWN_RELEASE;
}

/**
 * Resolve the release identity owned by the current API process.
 * Environment access is explicit so callers and tests cannot accidentally
 * confuse validation of a candidate with reading deployment configuration.
 */
export function resolveProcessReleaseIdentity(): string {
  return resolveReleaseIdentity(process.env.WOOF_RELEASE_SHA);
}
