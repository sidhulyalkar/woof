import { expect, test } from '@playwright/test';

test.describe('public Woof product truth', () => {
  test('synthetic demo explains the dogOS relationship loop before supporting applications', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/demo');

    await expect(page.getByRole('heading', { level: 1 })).toHaveText(
      'One useful next step, with memory and authority underneath it.'
    );
    await expect(page.getByText('A synthetic dogOS walkthrough', { exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: /Walk through Today/i })).toHaveAttribute(
      'href',
      '#today'
    );

    await expect(page.getByRole('heading', { name: 'A short sniff walk' })).toBeVisible();
    await expect(
      page.getByText(
        'Missing live context stays missing. Woof does not fill the gap with invented certainty.'
      )
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Story remembers the relationship, not a score.' })
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Temporary caregiver authority has edges.' })
    ).toBeVisible();

    await expect(page.getByText('Medical authority', { exact: true })).toBeVisible();
    await expect(page.getByText('Bond XP / reward authority', { exact: true })).toBeVisible();
    await expect(page.getByText('Not granted', { exact: true })).toHaveCount(4);

    await expect(page.getByRole('link', { name: /Explore compatibility/i })).toHaveCount(0);
    await expect(page.getByText('Synthetic beta demo', { exact: true })).toHaveCount(0);
  });
});
