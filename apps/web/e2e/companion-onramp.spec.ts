import { expect, test, type Page, type Route } from '@playwright/test';
import { seedAuthenticatedSession } from './support/session';

function corsHeaders(route: Route) {
  const requestHeaders = route.request().headers();
  return {
    'access-control-allow-origin': requestHeaders.origin ?? 'http://localhost:3000',
    'access-control-allow-methods': 'GET,POST,PUT,PATCH,DELETE,OPTIONS',
    'access-control-allow-headers':
      requestHeaders['access-control-request-headers'] ?? 'authorization,content-type',
    vary: 'Origin',
  };
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  if (route.request().method() === 'OPTIONS') {
    await route.fulfill({ status: 204, headers: corsHeaders(route), body: '' });
    return;
  }
  await route.fulfill({
    status,
    headers: { ...corsHeaders(route), 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

const browserUser = {
  id: 'ally-user',
  email: 'ally@example.com',
  handle: 'animal-ally',
  pets: [],
};

async function authenticate(page: Page) {
  await seedAuthenticatedSession(page, {
    user: browserUser,
    token: 'companion-browser-token',
  });
}

const allyState = {
  mode: 'ANIMAL_ALLY',
  modeSource: 'PERSISTED',
  hasAuthorizedPet: false,
  landing: 'COMPANION_TODAY',
  authority: {
    modeControlsPresentation: true,
    petAccessComesFromRelationships: true,
    modeNeverCreatesPetAuthority: true,
  },
};

async function mockPetlessSocial(page: Page) {
  await page.route('**/social-adventure/me', (route) =>
    fulfillJson(route, {
      preferences: { globalLeaderboardOptIn: false },
      season: {
        key: 'week:2026-08-24',
        startsAt: '2026-08-24T00:00:00.000Z',
        endsAt: '2026-08-31T00:00:00.000Z',
      },
      score: 50,
      maxScore: 375,
      components: {
        humanSkill: {
          score: 50,
          maxScore: 200,
          completedChallenges: ['MAKE_IT_EASIER'],
        },
        adventureVariety: { score: 0, maxScore: 175, pathways: [] },
      },
      humanSkillBestScores: { MAKE_IT_EASIER: 100 },
      policyVersion: 'social-adventure-score-v1',
      principles: ['human-skill-over-pet-performance'],
    })
  );
  await page.route('**/social-adventure/leaderboard/global**', (route) =>
    fulfillJson(route, {
      scope: 'GLOBAL',
      season: {
        key: 'week:2026-08-24',
        startsAt: '2026-08-24T00:00:00.000Z',
        endsAt: '2026-08-31T00:00:00.000Z',
      },
      entries: [],
      me: { score: 50, maxScore: 375, rank: null, public: false },
      policyVersion: 'social-adventure-score-v1',
      disclaimer: 'Human learning and bounded Adventure variety only.',
    })
  );
  await page.route('**/social-adventure/feed**', (route) =>
    fulfillJson(route, { posts: [], privacy: 'Public shares plus your own private shares.' })
  );
  await page.route('**/social-adventure/share-candidates**', (route) =>
    fulfillJson(route, { candidates: [], privacy: 'Nothing is posted automatically.' })
  );
}

test.describe('dogOS Companion Onramp', () => {
  test('authenticated returns resume at role choice instead of recreating account details', async ({
    page,
  }) => {
    await authenticate(page);
    await page.goto('/onboarding/companion');

    await expect(
      page.getByRole('heading', { name: 'You do not need a pet to start learning.' })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('Step 2 of 2')).toBeVisible();
    await expect(page.getByLabel('Email')).toHaveCount(0);
    await expect(page.getByLabel('Password')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Return to Woof' })).toBeVisible();
  });

  test('Animal Ally gets a useful petless Today and no pet-only navigation', async ({ page }) => {
    await authenticate(page);
    await page.route('**/companion/state', (route) => fulfillJson(route, allyState));
    await mockPetlessSocial(page);

    await page.goto('/');

    await expect(
      page.getByRole('heading', {
        name: 'Learn useful dog-human skills before you need a pet profile.',
      })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole('link', { name: 'Arcade', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Community', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Readiness', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Compass', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Story', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Auto', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Coach', exact: true })).toHaveCount(0);

    await page.goto('/community');
    await expect(page.getByRole('link', { name: /Skill Arcade/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Local Packs/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Cooperative Pack quests/i })).toHaveCount(0);
    await expect(page.getByText('Human learning league')).toBeVisible();
  });

  test('unavailable Companion authority fails closed instead of guessing guardian access', async ({
    page,
  }) => {
    await authenticate(page);
    await page.route('**/companion/state', (route) =>
      fulfillJson(route, { message: 'Companion state unavailable' }, 503)
    );

    await page.goto('/');

    await expect(
      page.getByRole('heading', { name: 'We could not resolve your Woof mode' })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole('link', { name: 'Arcade', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Community', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Readiness', exact: true })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Compass', exact: true })).toHaveCount(0);
    await expect(page.locator('[data-today-primary-quest]')).toHaveCount(0);
  });

  test('Pet Guardian mode without real pet authority stays in pet setup', async ({ page }) => {
    await authenticate(page);
    await page.route('**/companion/state', (route) =>
      fulfillJson(route, {
        mode: 'PET_GUARDIAN',
        modeSource: 'PERSISTED',
        hasAuthorizedPet: false,
        landing: 'NEEDS_PET_SETUP',
        authority: {
          modeControlsPresentation: true,
          petAccessComesFromRelationships: true,
          modeNeverCreatesPetAuthority: true,
        },
      })
    );

    await page.goto('/');

    await expect(
      page.getByRole('heading', { name: 'Add the dog you actually care for.' })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.getByRole('link', { name: 'Add a dog' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Compass', exact: true })).toHaveCount(0);
    await expect(page.locator('[data-today-primary-quest]')).toHaveCount(0);
  });
});
