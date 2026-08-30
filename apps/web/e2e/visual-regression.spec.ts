import { expect, test, type Page } from '@playwright/test';

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
}

test.describe('visual layout contracts', () => {
  test('login preserves desktop hierarchy', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.goto('/login');

    await expect(page.getByRole('heading', { name: 'Welcome back', level: 1 })).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
    await expectNoHorizontalOverflow(page);

    const main = page.locator('main').first();
    const box = await main.boundingBox();
    expect(box?.width ?? 0).toBeGreaterThan(500);
  });

  test('login remains usable at narrow mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/login');

    await expect(page.getByRole('heading', { name: 'Welcome back', level: 1 })).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
    await expectNoHorizontalOverflow(page);

    const interactiveElements = page.locator('button, input, a[href]');
    const count = await interactiveElements.count();
    for (let index = 0; index < count; index += 1) {
      const element = interactiveElements.nth(index);
      if (!(await element.isVisible())) continue;
      const box = await element.boundingBox();
      if (!box) continue;
      expect(box.height).toBeGreaterThanOrEqual(32);
    }
  });

  test('synthetic dogOS walkthrough keeps its evidence hierarchy stable', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto('/demo');

    await expect(
      page.getByRole('heading', {
        name: /one useful next step, with memory and authority underneath it/i,
      })
    ).toBeVisible();
    await expect(page.getByText('Synthetic data only')).toBeVisible();
    await expect(page.getByText('Context + provenance')).toBeVisible();
    await expect(
      page.getByText(/no live location, private messages, or real health records/i)
    ).toBeVisible();
    await expectNoHorizontalOverflow(page);

    const cards = page.locator('[data-demo-card]');
    expect(await cards.count()).toBeGreaterThanOrEqual(3);
  });
});
