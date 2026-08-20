import { expect, test } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('redirects unauthenticated visitors to login', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/.*login/);
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
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Invalid email or password' }),
      });
    });

    await page.goto('/login');
    await page.getByLabel(/email/i).fill('invalid@example.com');
    await page.getByLabel(/password/i).fill('wrongpassword');
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
