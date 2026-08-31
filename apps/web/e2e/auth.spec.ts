import { expect, test, type Route } from '@playwright/test';
import { E2E_ORIGIN } from './support/session';

function corsHeaders(route: Route) {
  const requestHeaders = route.request().headers();
  return {
    'access-control-allow-origin': requestHeaders.origin ?? E2E_ORIGIN,
    'access-control-allow-methods': 'POST,OPTIONS',
    'access-control-allow-headers':
      requestHeaders['access-control-request-headers'] ?? 'content-type',
    vary: 'Origin',
  };
}

test.describe('Authentication Flow', () => {
  test('redirects unauthenticated visitors to login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/.*login/);
  });

  test('binds the response CSP nonce to the Next runtime script', async ({ page }) => {
    const response = await page.goto('/login', { waitUntil: 'domcontentloaded' });
    if (!response) {
      throw new Error('Login navigation did not produce a document response.');
    }

    const csp = response.headers()['content-security-policy'];
    expect(csp).toContain("'strict-dynamic'");

    const nonceMatch = csp.match(/'nonce-([^']+)'/);
    const responseNonce = nonceMatch?.[1];
    if (!responseNonce) {
      throw new Error('Production CSP did not contain a request nonce.');
    }

    const runtimeScript = page.locator('script[src^="/_next/static/"]').first();
    const runtimeNonce = await runtimeScript.evaluate((script: HTMLScriptElement) => script.nonce);
    expect(runtimeNonce).toBe(responseNonce);
  });

  test('displays the login form accessibly', async ({ page }) => {
    await page.goto('/login');

    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    await expect(page.getByLabel(/email/i)).toBeVisible();
    await expect(page.getByLabel(/password/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
  });

  test('uses native validation for an empty form', async ({ page }) => {
    await page.goto('/login');
    await page.getByRole('button', { name: /sign in/i }).click();

    await expect(page.getByLabel(/email/i)).toBeFocused();
  });

  test('shows an API error without requiring a live backend', async ({ page }) => {
    await page.route('**/auth/login', async (route) => {
      if (route.request().method() === 'OPTIONS') {
        await route.fulfill({ status: 204, headers: corsHeaders(route), body: '' });
        return;
      }
      await route.fulfill({
        status: 401,
        headers: { ...corsHeaders(route), 'content-type': 'application/json' },
        body: JSON.stringify({ message: 'Invalid email or password' }),
      });
    });

    await page.goto('/login');
    const email = page.getByLabel(/email/i);
    const password = page.getByLabel(/password/i);
    await email.click();
    await email.pressSequentially('invalid@example.com');
    await password.click();
    await password.pressSequentially('wrongpassword');
    await expect(email).toHaveValue('invalid@example.com');
    await expect(password).toHaveValue('wrongpassword');
    await page.getByRole('button', { name: /sign in/i }).click();

    await expect(page.getByText(/invalid email or password/i)).toBeVisible({ timeout: 5000 });
  });

  test.skip('logs in with seeded credentials against the full local stack', async ({ page }) => {
    // Integration-only test: intentionally skipped unless the seeded API is running.
    await page.goto('/login');
    await page.getByLabel(/email/i).fill('demo@woof.com');
    await page.getByLabel(/password/i).fill('password123');
    await page.getByRole('button', { name: /sign in/i }).click();

    await expect(page).toHaveURL('/');
    await expect(page.getByText(/woof/i).first()).toBeVisible();
  });
});
