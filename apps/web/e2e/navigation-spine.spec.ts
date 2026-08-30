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

const pet = {
  id: 'nav-pet-1',
  name: 'Nova',
  species: 'DOG',
  breed: 'Husky mix',
  avatarUrl: null,
  createdAt: '2026-01-01T00:00:00.000Z',
  _count: { activities: 3, posts: 0 },
};

const owner = {
  id: 'nav-owner-1',
  email: 'nav-owner@example.com',
  handle: 'nav-owner',
  pets: [pet],
};

async function authenticate(page: Page, user = owner) {
  await seedAuthenticatedSession(page, {
    user,
    token: 'navigation-browser-token',
  });
}

function authorizePetToday(page: Page) {
  return page.route('**/companion/state', (route) =>
    fulfillJson(route, {
      mode: 'PET_GUARDIAN',
      modeSource: 'PERSISTED',
      hasAuthorizedPet: true,
      landing: 'PET_TODAY',
      authority: {
        modeControlsPresentation: true,
        petAccessComesFromRelationships: true,
        modeNeverCreatesPetAuthority: true,
      },
    })
  );
}

async function mockGuardianToday(page: Page) {
  await page.route('**/pets/me**', (route) =>
    fulfillJson(route, { pets: [pet], total: 1, skip: 0, take: 100 })
  );
  await page.route('**/adventure/me**', (route) =>
    fulfillJson(route, {
      pet: { id: pet.id, name: pet.name, species: pet.species, avatarUrl: null },
      generatedAt: '2026-08-30T06:00:00.000Z',
      bondXp: 12,
      rhythm: { activeWeeks: 1, windowWeeks: 4, label: 'Finding your rhythm' },
      compass: [],
      quests: [
        {
          id: 'nav-quest-1',
          key: 'nav-sniff-walk',
          title: 'Neighborhood sniff walk',
          description: 'A low-pressure option for today.',
          why: 'Recent context makes an easy exploratory walk a reasonable choice.',
          primaryPathway: 'EXPLORE',
          pathways: ['EXPLORE'],
          xp: 8,
          confidence: 0.7,
          href: '/activity',
          actionLabel: 'Open activity',
          variant: 'recommended',
          safeStopEligible: true,
          personalRelevance: 0.8,
          expiresAt: '2026-08-31T06:00:00.000Z',
        },
      ],
      learningSummary: [],
      principles: [],
      disclaimer: 'Recent opportunity coverage, not a health score.',
    })
  );
  await page.route('**/concierge/today**', (route) =>
    fulfillJson(route, {
      generatedAt: '2026-08-30T06:00:00.000Z',
      pet: { id: pet.id, name: pet.name, species: pet.species, avatarUrl: null },
      briefing: {
        title: "Nova's day at a glance",
        summary: 'No urgent care context is surfaced right now.',
        topQuest: null,
      },
      context: {
        weather: { status: 'NOT_CONFIGURED', live: false, detail: 'No live weather.' },
        pace: { mode: 'NORMAL', reason: 'No lower-pace feedback.', evidence: [] },
      },
      suggestions: [],
      connectorSummary: { connected: 0, needsReauthorization: 0 },
      boundaries: {
        suggestionOnly: true,
        liveWeatherUsed: false,
        diagnosticInferenceAllowed: false,
        prescriptionOrDoseCalculationAllowed: false,
        persistentStateMutationAllowed: false,
        autonomousPurchaseAllowed: false,
      },
    })
  );
}

async function mockCommunity(page: Page) {
  await page.route('**/social-adventure/me', (route) =>
    fulfillJson(route, {
      preferences: { globalLeaderboardOptIn: false },
      season: {
        key: 'week:2026-08-24',
        startsAt: '2026-08-24T00:00:00.000Z',
        endsAt: '2026-08-31T00:00:00.000Z',
      },
      score: 0,
      maxScore: 375,
      components: {
        humanSkill: { score: 0, maxScore: 200, completedChallenges: [] },
        adventureVariety: { score: 0, maxScore: 175, pathways: [] },
      },
      humanSkillBestScores: {},
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
      me: { score: 0, maxScore: 375, rank: null, public: false },
      policyVersion: 'social-adventure-score-v1',
      disclaimer: 'Human learning and bounded Adventure variety only.',
    })
  );
  await page.route('**/social-adventure/feed**', (route) =>
    fulfillJson(route, { posts: [], privacy: 'Nothing is posted automatically.' })
  );
  await page.route('**/social-adventure/share-candidates**', (route) =>
    fulfillJson(route, { candidates: [], privacy: 'Nothing is posted automatically.' })
  );
}

test.describe('dogOS navigation spine', () => {
  test('pet mode keeps four primary jobs and makes deeper tools contextual on Today', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await authenticate(page);
    await authorizePetToday(page);
    await mockGuardianToday(page);

    await page.goto('/');

    const nav = page.getByRole('navigation', { name: 'Primary navigation' });
    await expect(nav.getByRole('link')).toHaveCount(4);
    await expect(nav.getByRole('link', { name: 'Today', exact: true })).toHaveAttribute(
      'aria-current',
      'page'
    );
    await expect(nav.getByRole('link', { name: 'Compass', exact: true })).toBeVisible();
    await expect(nav.getByRole('link', { name: 'Open Our Story' })).toBeVisible();
    await expect(nav.getByRole('link', { name: 'Community', exact: true })).toBeVisible();
    await expect(nav.getByRole('link', { name: 'Auto', exact: true })).toHaveCount(0);
    await expect(nav.getByRole('link', { name: 'Coach', exact: true })).toHaveCount(0);

    const tools = page.locator('[data-today-tools]');
    await expect(tools.getByRole('link', { name: /Practice with Coach/i })).toHaveAttribute(
      'href',
      '/coach'
    );
    await expect(tools.getByRole('link', { name: /Reminders & check-ins/i })).toHaveAttribute(
      'href',
      '/autopilot'
    );
  });

  test('Community owns pet discovery without changing the four-part primary spine', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await authenticate(page);
    await authorizePetToday(page);
    await mockCommunity(page);

    await page.goto('/community');

    const nav = page.getByRole('navigation', { name: 'Primary navigation' });
    await expect(nav.getByRole('link')).toHaveCount(4);
    await expect(nav.getByRole('link', { name: 'Community', exact: true })).toHaveAttribute(
      'aria-current',
      'page'
    );
    await expect(
      page.getByRole('link', { name: /Discover compatible dogs & places/i })
    ).toHaveAttribute('href', '/discover');
  });
});
