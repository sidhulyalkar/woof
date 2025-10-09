import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test('should navigate to login page', async ({ page }) => {
    await page.goto('/');

    // Should redirect to login if not authenticated
    await expect(page).toHaveURL(/.*login/);
  });

  test('should display login form', async ({ page }) => {
    await page.goto('/login');

    await expect(page.getByRole('heading', { name: /welcome back/i })).toBeVisible();
    await expect(page.getByLabelText(/email/i)).toBeVisible();
    await expect(page.getByLabelText(/password/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
  });

  test('should show validation error for empty form', async ({ page }) => {
    await page.goto('/login');

    await page.getByRole('button', { name: /sign in/i }).click();

    // Browser's built-in validation should prevent form submission
    const emailInput = page.getByLabelText(/email/i);
    await expect(emailInput).toBeFocused();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.getByLabelText(/email/i).fill('invalid@example.com');
    await page.getByLabelText(/password/i).fill('wrongpassword');
    await page.getByRole('button', { name: /sign in/i }).click();

    // Should show error message (timing may vary)
    await expect(page.getByText(/invalid email or password/i)).toBeVisible({ timeout: 5000 });
  });

  test.skip('should login successfully with valid credentials', async ({ page }) => {
    // This test requires a running backend with seed data
    await page.goto('/login');

    await page.getByLabelText(/email/i).fill('demo@woof.com');
    await page.getByLabelText(/password/i).fill('password123');
    await page.getByRole('button', { name: /sign in/i }).click();

    // Should redirect to home page after successful login
    await expect(page).toHaveURL('/');
    await expect(page.getByText(/PetPath/i)).toBeVisible();
  });
});
