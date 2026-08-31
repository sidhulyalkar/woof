import { expect, test, type Route } from '@playwright/test';
import { E2E_ORIGIN, seedAuthenticatedSession, type AuthUser } from './support/session';

function corsHeaders(route: Route) {
  const requestHeaders = route.request().headers();
  return {
    'access-control-allow-origin': requestHeaders.origin ?? E2E_ORIGIN,
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

async function authenticate(page: Parameters<typeof seedAuthenticatedSession>[0], user: AuthUser) {
  await seedAuthenticatedSession(page, { user, token: 'browser-test-token' });
}

async function authorizePetToday(page: Parameters<typeof seedAuthenticatedSession>[0]) {
  await page.route('**/companion/state', (route) =>
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

const pets = [
  {
    id: 'pet-1',
    name: 'Nova',
    species: 'DOG',
    breed: 'Husky mix',
    avatarUrl: null,
    createdAt: '2025-01-01T00:00:00.000Z',
    _count: { activities: 8, posts: 1 },
  },
  {
    id: 'pet-2',
    name: 'Miso',
    species: 'DOG',
    breed: 'Senior mix',
    avatarUrl: null,
    createdAt: '2025-02-01T00:00:00.000Z',
    _count: { activities: 3, posts: 0 },
  },
];

const pixel = {
  id: 'pet-new',
  name: 'Pixel',
  species: 'DOG',
  breed: 'Mix',
  avatarUrl: null,
  createdAt: '2026-08-22T20:00:00.000Z',
  _count: { activities: 0, posts: 0 },
};

const guardianUser: AuthUser = {
  id: 'user-1',
  email: 'owner@example.com',
  handle: 'dog-owner',
  pets: pets.map(({ id, name, species, breed, avatarUrl }) => ({
    id,
    name,
    species,
    breed,
    avatarUrl,
  })),
};

const petlessUser: AuthUser = {
  id: guardianUser.id,
  email: guardianUser.email,
  handle: guardianUser.handle,
  pets: [],
};

function productPet(petId: string) {
  return [...pets, pixel].find((candidate) => candidate.id === petId) ?? pets[0]!;
}

function dashboard(petId: string) {
  const pet = productPet(petId);
  const recovery = pet.id === 'pet-2';
  return {
    pet: { id: pet.id, name: pet.name, species: pet.species, avatarUrl: null },
    generatedAt: `2026-08-22T20:00:${pet.id === 'pet-1' ? '01' : pet.id === 'pet-2' ? '02' : '03'}.000Z`,
    bondXp: pet.id === 'pet-1' ? 42 : pet.id === 'pet-2' ? 18 : 0,
    rhythm: { activeWeeks: 2, windowWeeks: 4, label: 'Finding your rhythm' },
    compass: [],
    quests: [
      {
        id: `quest-${pet.id}`,
        key: `walk-${pet.id}`,
        title: recovery ? 'Easy porch decompression' : 'Neighborhood sniff walk',
        description: 'A low-pressure option for today.',
        why: `Chosen from recent context for ${pet.name}.`,
        primaryPathway: recovery ? 'RECOVER' : 'MOVE',
        pathways: [recovery ? 'RECOVER' : 'MOVE'],
        xp: 8,
        confidence: 0.7,
        href: '/activity',
        actionLabel: 'Open activity',
        variant: 'recommended',
        safeStopEligible: true,
        personalRelevance: 0.8,
        expiresAt: '2026-08-23T20:00:00.000Z',
      },
    ],
    learningSummary: [],
    principles: [],
    disclaimer: 'Recent opportunity coverage, not a health score.',
  };
}

function concierge(petId: string) {
  const pet = productPet(petId);
  return {
    generatedAt: '2026-08-22T20:00:00.000Z',
    pet: { id: pet.id, name: pet.name, species: pet.species, avatarUrl: null },
    briefing: {
      title: `${pet.name}'s day at a glance`,
      summary: 'No urgent care context is surfaced right now.',
      topQuest: {
        title: dashboard(pet.id).quests[0]!.title,
        reason: `Chosen from recent context for ${pet.name}.`,
        action: { label: 'Open activity', href: '/activity' },
        evidence: [{ source: 'ADVENTURE', label: 'Adventure ranking.' }],
      },
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
  };
}

test.describe('dogOS release polish', () => {
  test('Today leads with one recommendation while keeping Concierge and the selected dog aligned', async ({
    page,
  }) => {
    await authenticate(page, guardianUser);
    await authorizePetToday(page);
    await page.route('**/pets/me**', (route) =>
      fulfillJson(route, { pets, total: 2, skip: 0, take: 100 })
    );
    await page.route('**/adventure/me**', (route) => {
      const petId = new URL(route.request().url()).searchParams.get('petId') ?? 'pet-1';
      return fulfillJson(route, dashboard(petId));
    });
    await page.route('**/concierge/today**', (route) => {
      const petId = new URL(route.request().url()).searchParams.get('petId') ?? 'pet-1';
      return fulfillJson(route, concierge(petId));
    });

    await page.goto('/');

    const primaryQuest = page.locator('[data-today-primary-quest]');
    const conciergeDetails = page.locator('[data-today-concierge]');
    const progress = page.locator('[data-today-progress]');

    await expect(primaryQuest).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Neighborhood sniff walk', level: 1 })
    ).toBeVisible();
    await expect(primaryQuest.getByText(/Why this one today/i)).toBeVisible();
    await expect(primaryQuest.getByRole('button', { name: 'Open activity' })).toBeVisible();
    await expect(page.getByRole('heading', { name: /Nova's day at a glance/i })).toHaveCount(0);
    expect(await conciergeDetails.getAttribute('open')).toBeNull();

    const primaryBox = await primaryQuest.boundingBox();
    const conciergeBox = await conciergeDetails.boundingBox();
    const progressBox = await progress.boundingBox();
    expect(primaryBox).not.toBeNull();
    expect(conciergeBox).not.toBeNull();
    expect(progressBox).not.toBeNull();
    expect(primaryBox!.y).toBeLessThan(conciergeBox!.y);
    expect(primaryBox!.y).toBeLessThan(progressBox!.y);

    await page.getByText(/More context for today/i).click();
    await expect(page.getByRole('heading', { name: /Nova's day at a glance/i })).toBeVisible();

    await page.getByRole('button', { name: 'Miso' }).click();

    await expect(page).toHaveURL(/pet=pet-2/);
    await expect(
      page.getByRole('heading', { name: 'Easy porch decompression', level: 1 })
    ).toBeVisible();
    await expect(page.getByRole('heading', { name: /Miso's day at a glance/i })).toBeVisible();
  });

  test('Activity reads canonical history, switches dogs, and quick-logs without fake route data', async ({
    page,
  }) => {
    await authenticate(page, guardianUser);
    await authorizePetToday(page);
    await page.route('**/pets/me**', (route) =>
      fulfillJson(route, { pets, total: 2, skip: 0, take: 100 })
    );

    let createdBody: Record<string, unknown> | null = null;
    await page.route('**/activities**', async (route) => {
      if (route.request().method() === 'OPTIONS') return fulfillJson(route, {});
      if (route.request().method() === 'POST') {
        createdBody = route.request().postDataJSON() as Record<string, unknown>;
        return fulfillJson(route, { id: 'activity-new', ...createdBody }, 201);
      }
      const petId = new URL(route.request().url()).searchParams.get('petId') ?? 'pet-1';
      const name = petId === 'pet-2' ? 'Miso' : 'Nova';
      return fulfillJson(route, {
        activities: [
          {
            id: `activity-${petId}`,
            userId: 'user-1',
            petId,
            householdId: 'house-1',
            startedAt: '2026-08-22T18:00:00.000Z',
            endedAt: '2026-08-22T18:30:00.000Z',
            type: petId === 'pet-2' ? 'RECOVERY' : 'WALK',
            route: null,
            petParticipants: [
              {
                petId,
                metrics: null,
                pet: { id: petId, name, species: 'DOG', avatarUrl: null },
              },
            ],
          },
        ],
        total: 1,
        skip: 0,
        take: 20,
      });
    });

    await page.goto('/activity');
    const novaHistory = page.getByRole('region', { name: /Nova's history/i });
    await expect(novaHistory).toBeVisible({ timeout: 10_000 });
    await expect(novaHistory.getByText('Walk', { exact: true })).toBeVisible();
    await expect(page.getByText(/Great walk in the park/i)).toHaveCount(0);

    await page.getByRole('button', { name: 'Miso' }).click();
    const misoHistory = page.getByRole('region', { name: /Miso's history/i });
    await expect(misoHistory).toBeVisible({ timeout: 10_000 });
    await expect(misoHistory.getByText('Recovery', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Play' }).click();
    await page.getByRole('button', { name: '15m' }).click();
    await page.getByRole('button', { name: /Save 15 min play/i }).click();

    await expect(page.getByText(/Play saved for Miso/i)).toBeVisible();
    expect(createdBody).toMatchObject({
      petIds: ['pet-2'],
      type: 'PLAY',
      jointMetrics: { source: 'MANUAL_QUICK_LOG', enteredDurationMinutes: 15 },
    });
    expect(createdBody).not.toHaveProperty('route');
    expect(createdBody).not.toHaveProperty('bondXp');
  });

  test('first-dog onboarding writes the minimum profile and makes it active', async ({ page }) => {
    await authenticate(page, petlessUser);
    let createdBody: Record<string, unknown> | null = null;
    let created = false;
    await page.route('**/companion/state', (route) =>
      fulfillJson(route, {
        mode: 'PET_GUARDIAN',
        modeSource: 'PERSISTED',
        hasAuthorizedPet: created,
        landing: created ? 'PET_TODAY' : 'NEEDS_PET_SETUP',
        authority: {
          modeControlsPresentation: true,
          petAccessComesFromRelationships: true,
          modeNeverCreatesPetAuthority: true,
        },
      })
    );
    await page.route('**/pets', async (route) => {
      if (route.request().method() === 'OPTIONS') return fulfillJson(route, {});
      createdBody = route.request().postDataJSON() as Record<string, unknown>;
      created = true;
      return fulfillJson(route, pixel, 201);
    });
    await page.route('**/pets/me**', (route) =>
      fulfillJson(route, {
        pets: created ? [pixel] : [],
        total: created ? 1 : 0,
        skip: 0,
        take: 100,
      })
    );
    await page.route('**/adventure/me**', (route) => {
      const petId = new URL(route.request().url()).searchParams.get('petId') ?? 'pet-new';
      return fulfillJson(route, dashboard(petId));
    });
    await page.route('**/concierge/today**', (route) => {
      const petId = new URL(route.request().url()).searchParams.get('petId') ?? 'pet-new';
      return fulfillJson(route, concierge(petId));
    });

    await page.goto('/pets/new');
    await page.getByLabel('Name').fill('Pixel');
    await page.getByLabel(/Breed or mix/i).fill('Mix');
    await page.getByRole('button', { name: /Meet Pixel/i }).click();

    await expect(page).toHaveURL(/\/?pet=pet-new/);
    await expect(
      page.getByRole('heading', { name: 'Neighborhood sniff walk', level: 1 })
    ).toBeVisible();
    expect(createdBody).toMatchObject({ name: 'Pixel', species: 'DOG', breed: 'Mix' });
    expect(createdBody).not.toHaveProperty('temperament');
    expect(createdBody).not.toHaveProperty('vaccinations');
  });
});
