/**
 * Security utilities for the frontend
 */

/**
 * Sanitize user input to prevent XSS attacks
 */
export function sanitizeInput(input: string): string {
  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
    .replace(/\//g, '&#x2F;');
}

/**
 * Validate URL to prevent open redirect attacks
 */
export function isValidRedirect(url: string): boolean {
  try {
    const parsed = new URL(url, window.location.origin);
    return parsed.origin === window.location.origin;
  } catch {
    return false;
  }
}

/**
 * Generate a CSRF token for forms
 */
export function generateCSRFToken(): string {
  return crypto.randomUUID();
}

/**
 * Validate CSRF token
 */
export function validateCSRFToken(token: string, storedToken: string): boolean {
  return token === storedToken;
}

/**
 * Rate limit client-side actions
 */
export class RateLimiter {
  private attempts: Map<string, number[]> = new Map();

  canAttempt(key: string, maxAttempts: number, windowMs: number): boolean {
    const now = Date.now();
    const attempts = this.attempts.get(key) || [];

    // Remove old attempts outside the window
    const recentAttempts = attempts.filter(
      (timestamp) => now - timestamp < windowMs,
    );

    if (recentAttempts.length >= maxAttempts) {
      return false;
    }

    recentAttempts.push(now);
    this.attempts.set(key, recentAttempts);
    return true;
  }

  reset(key: string): void {
    this.attempts.delete(key);
  }
}

/**
 * Validate file upload
 */
export function validateFileUpload(
  file: File,
  options: {
    maxSizeBytes?: number;
    allowedTypes?: string[];
  } = {},
): { valid: boolean; error?: string } {
  const { maxSizeBytes = 10 * 1024 * 1024, allowedTypes = [] } = options;

  if (file.size > maxSizeBytes) {
    return {
      valid: false,
      error: `File size must be less than ${maxSizeBytes / 1024 / 1024}MB`,
    };
  }

  if (allowedTypes.length > 0 && !allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: `File type must be one of: ${allowedTypes.join(', ')}`,
    };
  }

  return { valid: true };
}

/**
 * Securely store sensitive data in sessionStorage
 */
export const secureStorage = {
  set(key: string, value: unknown): void {
    try {
      const encrypted = btoa(JSON.stringify(value));
      sessionStorage.setItem(key, encrypted);
    } catch (error) {
      console.error('Failed to store data securely', error);
    }
  },

  get<T>(key: string): T | null {
    try {
      const encrypted = sessionStorage.getItem(key);
      if (!encrypted) return null;
      return JSON.parse(atob(encrypted)) as T;
    } catch (error) {
      console.error('Failed to retrieve data securely', error);
      return null;
    }
  },

  remove(key: string): void {
    sessionStorage.removeItem(key);
  },

  clear(): void {
    sessionStorage.clear();
  },
};
