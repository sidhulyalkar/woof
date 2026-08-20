import { expect, test } from '@playwright/test';

async function expectNoHorizontalOverflow(page: Parameters<typeof test>[0] extends never ? never : any) {
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
    await page.waitForLoadState('networkidle');

    await expect(page.getByRole('heading', { name: /welcome|woof/i }).first()).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in|log in/i })).toBeVisible();
    await expectNoHorizontalOverflow(page);

    const main = page.locator('main').first();
    if (await main.count()) {
      const box = await main.boundingBox();
      expect(box?.width ?? 0).toBeGreaterThan(700);
    }
  });

  test('login remains usable at narrow mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/login');
    await page.waitForLoadState('networkidle');

    await expect(page.getByRole('button', { name: /sign in|log in/i })).toBeVisible();
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

  test('synthetic demo keeps its evidence hierarchy stable', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto('/demo');
    await page.waitForLoadState('networkidle');

    await expect(page.getByRole('heading', { name: /synthetic beta demo/i })).toBeVisible();
    await expect(page.getByText(/compatibility/i).first()).toBeVisible();
    await expect(page.getByText(/no live location/i)).toBeVisible();
    await expectNoHorizontalOverflow(page);

    const cards = page.locator('[data-demo-card]');
    expect(await cards.count()).toBeGreaterThanOrEqual(3);
  });
});
