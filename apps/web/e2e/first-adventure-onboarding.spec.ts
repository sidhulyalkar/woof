import { expect, test, type Page } from '@playwright/test';

const apiUser = {
  id: 'user-first-adventure',
  handle: 'trailpaws',
  email: 'trailpaws@example.com',
  bio: null,
  avatarUrl: null,
  isVerified: false,
  pets: [],
};

const createdPet = {
  id: 'pet-first-adventure',
  name: 'Mochi',
  species: 'DOG',
  breed: 'Mix',
  birthdate: '2023-05-01T00:00:00.000Z',
  avatarUrl: null,
  createdAt: '2026-08-26T00:00:00.000Z',
  householdMemberships: [{ householdId: 'house-first-adventure' }],
};

async function fillOwner(page: Page) {
  const handle = page.getByLabel(/public handle/i);
  const email = page.getByLabel(/^email$/i);
  const password = page.getByLabel(/^password$/i);
  const continueButton = page.getByRole('button', { name: /continue to pet profile/i });

  // The onboarding page is server-rendered. On a hot retry, Playwright can reach
  // the inputs before React has attached controlled-input handlers. Prove the
  // client state owns the values before continuing rather than relying on timing.
  for (let attempt = 0; attempt < 3; attempt += 1) {
    await handle.fill('trailpaws');
    await email.fill('trailpaws@example.com');
    await password.fill('relationship123');

    try {
      await expect(continueButton).toBeEnabled({ timeout: 1_500 });
      break;
    } catch (error) {
      if (attempt === 2) throw error;
    }
  }

  await continueButton.click();
}

async function fillPet(page: Page) {
  await page.getByLabel(/pet name/i).fill('Mochi');
  await page.getByLabel(/birthday or best estimate/i).fill('2023-05-01');
  await page.getByLabel(/breed or mix/i).fill('Mix');
  await page.getByRole('button', { name: /continue to first adventure/i }).click();
}

test.describe('relationship-first First Adventure onboarding', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/auth/register', async (route) => {
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({ access_token: 'first-adventure-token', user: apiUser }),
      });
    });

    await page.route('**/auth/me', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...apiUser, pets: [createdPet] }),
      });
    });
  });

  test('creates the durable pair before optional personalization and never revives the legacy quiz', async ({
    page,
  }) => {
    const petCreateBodies: Array<Record<string, unknown>> = [];
    const profileWrites: Array<Record<string, unknown>> = [];

    await page.route('**/pets', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fallback();
        return;
      }
      petCreateBodies.push(route.request().postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify(createdPet),
      });
    });

    await page.route('**/adventure/profile/**/questions/respond', async (route) => {
      profileWrites.push(route.request().postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'Optional profile store unavailable' }),
      });
    });

    await page.goto('/onboarding');
    await fillOwner(page);
    await fillPet(page);

    await expect(page.getByRole('heading', { name: /make the first suggestion/i })).toBeVisible();
    await expect(page.getByText(/matching preferences/i)).toHaveCount(0);
    expect(petCreateBodies).toHaveLength(1);

    const createBody = petCreateBodies[0]!;
    expect(createBody).toMatchObject({
      name: 'Mochi',
      species: 'DOG',
      breed: 'Mix',
      birthdate: '2023-05-01',
    });
    expect(createBody.creationKey).toMatch(/^first-adventure:/);
    expect(createBody).not.toHaveProperty('temperament');
    expect(createBody).not.toHaveProperty('avatarUrl');

    await page.getByRole('button', { name: /skip personalization for now/i }).click();

    await expect(page).toHaveURL(/\/$/);
    expect(profileWrites.length).toBeGreaterThan(0);
    expect(profileWrites.every((write) => write.outcome === 'SKIPPED')).toBe(true);
  });

  test('reuses the registration transaction key after a lost response before pet creation', async ({
    page,
  }) => {
    await page.unroute('**/auth/register');
    const registrationBodies: Array<Record<string, unknown>> = [];
    let petCreateCount = 0;

    await page.route('**/auth/register', async (route) => {
      registrationBodies.push(route.request().postDataJSON() as Record<string, unknown>);
      if (registrationBodies.length === 1) {
        await route.abort('failed');
        return;
      }

      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify({ access_token: 'recovered-registration-token', user: apiUser }),
      });
    });

    await page.route('**/pets', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fallback();
        return;
      }
      petCreateCount += 1;
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify(createdPet),
      });
    });

    await page.goto('/onboarding');
    await fillOwner(page);
    await fillPet(page);

    await expect(page.locator('#main-content [role="alert"]')).toContainText(/transaction keys/i);
    expect(registrationBodies).toHaveLength(1);
    expect(petCreateCount).toBe(0);

    await page.getByRole('button', { name: /continue to first adventure/i }).click();
    await expect(page.getByRole('heading', { name: /make the first suggestion/i })).toBeVisible();

    expect(registrationBodies).toHaveLength(2);
    expect(petCreateCount).toBe(1);
    const firstKey = registrationBodies[0]!.registrationKey;
    const secondKey = registrationBodies[1]!.registrationKey;
    expect(firstKey).toEqual(expect.any(String));
    expect(firstKey).toMatch(/^[0-9a-f-]{36}$/i);
    expect(secondKey).toBe(firstKey);
  });

  test('editing from First Adventure updates the same pet instead of creating another one', async ({
    page,
  }) => {
    let createCount = 0;
    const updates: Array<Record<string, unknown>> = [];

    await page.route('**/pets', async (route) => {
      if (route.request().method() !== 'POST') {
        await route.fallback();
        return;
      }
      createCount += 1;
      await route.fulfill({
        status: 201,
        contentType: 'application/json',
        body: JSON.stringify(createdPet),
      });
    });

    await page.route('**/pets/pet-first-adventure', async (route) => {
      if (route.request().method() !== 'PUT') {
        await route.fallback();
        return;
      }
      updates.push(route.request().postDataJSON() as Record<string, unknown>);
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ...createdPet, breed: null }),
      });
    });

    await page.goto('/onboarding');
    await fillOwner(page);
    await fillPet(page);
    await expect(page.getByRole('heading', { name: /make the first suggestion/i })).toBeVisible();

    await page.getByRole('button', { name: /review pet details/i }).click();
    await page.getByLabel(/breed or mix/i).fill('');
    await page.getByRole('button', { name: /continue to first adventure/i }).click();

    await expect(page.getByRole('heading', { name: /make the first suggestion/i })).toBeVisible();
    expect(createCount).toBe(1);
    expect(updates).toHaveLength(1);
    expect(updates[0]).toMatchObject({
      name: 'Mochi',
      species: 'DOG',
      breed: '',
      birthdate: '2023-05-01',
    });
  });
});
