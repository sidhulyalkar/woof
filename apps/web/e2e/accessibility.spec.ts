import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const publicRoutes = ['/login', '/onboarding', '/demo'];

for (const route of publicRoutes) {
  test(`${route} has no serious WCAG A/AA violations`, async ({ page }) => {
    await page.goto(route);
    await page.waitForLoadState('networkidle');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();

    const blockingViolations = results.violations.filter((violation) =>
      ['serious', 'critical'].includes(violation.impact ?? ''),
    );

    expect(
      blockingViolations,
      blockingViolations
        .map(
          (violation) =>
            `${violation.id}: ${violation.help}\n${violation.nodes
              .map(
                (node) =>
                  `  ${node.target.join(' ')}: ${node.failureSummary ?? ''}`,
              )
              .join('\n')}`,
        )
        .join('\n\n'),
    ).toEqual([]);
  });
}

test('public beta surfaces remain keyboard navigable', async ({ page }) => {
  await page.goto('/login');
  await page.keyboard.press('Tab');

  const firstFocused = await page.evaluate(
    () => document.activeElement?.tagName,
  );
  expect(firstFocused).not.toBe('BODY');

  await page.keyboard.press('Tab');
  const secondFocused = await page.evaluate(
    () => document.activeElement?.tagName,
  );
  expect(secondFocused).not.toBe('BODY');
});
