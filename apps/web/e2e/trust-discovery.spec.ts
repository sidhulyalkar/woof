import { expect, test, type Page, type Route } from '@playwright/test';

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

const user = {
  id: 'user-1',
  email: 'owner@example.com',
  handle: 'nova-human',
  pets: [
    {
      id: 'pet-1',
      name: 'Nova',
      species: 'DOG',
      breed: 'Husky mix',
      avatarUrl: null,
    },
  ],
};

async function authenticate(page: Page) {
  const token = 'trust-browser-token';

  // Model geolocation through Playwright's browser context rather than replacing
  // navigator.geolocation with an injected inline script. This keeps the test
  // compatible with the same CSP that protects the production app.
  await page.context().grantPermissions(['geolocation'], { origin: 'http://localhost:3000' });
  await page.context().setGeolocation({ latitude: 37.7749, longitude: -122.4194, accuracy: 100 });

  // Persist the real client auth shape before protected navigation so AuthGuard
  // does not race through session hydration. Discovery still performs its own
  // canonical profile refresh below, because current pet membership is part of
  // that feature's freshness contract rather than authentication bootstrap.
  await page.goto('/login');
  await page.evaluate(
    ({ token, user }) => {
      window.localStorage.setItem('authToken', token);
      window.localStorage.setItem(
        'woof-auth-storage',
        JSON.stringify({
          state: { user, token, isAuthenticated: true },
          version: 0,
        })
      );
    },
    { token, user }
  );
  await page.route('**/auth/me', (route) => fulfillJson(route, user));
}

const recommendation = {
  id: 'candidate-pet-2',
  pet: {
    id: 'pet-2',
    ownerId: 'user-2',
    name: 'Luna',
    species: 'DOG',
    breed: 'Retriever mix',
    birthdate: null,
    avatarUrl: null,
    temperament: ['calm'],
    owner: {
      id: 'user-2',
      handle: 'luna-human',
      bio: 'Likes quiet parallel walks.',
      avatarUrl: null,
      isVerified: true,
    },
  },
  compatibilityScore: 0.87,
  confidence: 0.78,
  source: 'behavior-outcome-baseline-v2',
  factors: { species: 1, temperament: 0.82 },
  explanation: ['Similar recent social pace'],
  status: 'PROPOSED',
  lastInteractionAt: null,
};

test('explicit rough-location discovery leads to an empty canonical conversation, never mock state', async ({
  page,
}) => {
  await authenticate(page);
  let locationEnabled = false;
  let conversationCreated = false;

  await page.route('**/compatibility/recommendations/pet-1**', (route) =>
    fulfillJson(route, { recommendations: [recommendation] })
  );
  await page.route('**/discovery/location', async (route) => {
    if (route.request().method() === 'PUT') {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      expect(body).toEqual({ latitude: 37.7749, longitude: -122.4194 });
      locationEnabled = true;
      return fulfillJson(route, {
        status: 'OPTED_IN',
        exactLocationStored: false,
        precisionMeters: 2200,
        expiresAt: '2026-09-22T08:00:00.000Z',
      });
    }
    return fulfillJson(route, {
      status: locationEnabled ? 'OPTED_IN' : 'NOT_CONFIGURED',
      exactLocationStored: false,
      precisionMeters: 2200,
    });
  });
  await page.route('**/discovery/nearby/pet-1**', (route) =>
    fulfillJson(route, {
      petId: 'pet-1',
      locationStatus: 'OPTED_IN',
      candidates: [
        {
          petId: 'pet-2',
          ownerId: 'user-2',
          petName: 'Luna',
          species: 'DOG',
          breed: 'Retriever mix',
          avatarUrl: null,
          owner: {
            id: 'user-2',
            handle: 'luna-human',
            avatarUrl: null,
            isVerified: true,
          },
          distanceBand: 'WITHIN_5_KM',
        },
      ],
      boundaries: {
        exactCoordinatesStored: false,
        exactCoordinatesReturned: false,
        homeLocationExposed: false,
        blockedUsersExcluded: true,
        publicProfilesOnly: true,
        maxRadiusKm: 10,
      },
    })
  );

  await page.route('**/chat/conversations', async (route) => {
    if (route.request().method() === 'POST') {
      expect(route.request().postDataJSON()).toEqual({ participantId: 'user-2' });
      conversationCreated = true;
      return fulfillJson(route, { id: 'conversation-1', created: true }, 201);
    }
    return fulfillJson(
      route,
      conversationCreated
        ? [
            {
              id: 'conversation-1',
              participant: {
                id: 'user-2',
                name: 'luna-human',
                avatarUrl: null,
                petId: 'pet-2',
                petName: 'Luna',
                petAvatarUrl: null,
              },
              lastMessage: null,
              unreadCount: 0,
              updatedAt: '2026-08-23T08:00:00.000Z',
            },
          ]
        : []
    );
  });
  await page.route('**/chat/conversations/conversation-1/messages**', (route) =>
    fulfillJson(route, { data: [], total: 0, page: 1, limit: 50 })
  );
  await page.route('**/chat/conversations/conversation-1/read', (route) =>
    fulfillJson(route, { ok: true })
  );

  await page.goto('/discover');
  await expect(page.getByRole('heading', { name: /Luna with luna-human/i })).toBeVisible();
  await expect(page.getByText(/within about 5 km/i)).toHaveCount(0);

  await page.getByRole('button', { name: /Use rough location/i }).click();
  await expect(page.getByText(/within about 5 km/i)).toBeVisible();
  await expect(page.getByText(/never receive your coordinates or home location/i)).toBeVisible();

  await page.getByRole('link', { name: /Start a conversation/i }).click();
  await expect(page).toHaveURL(/inbox\?member=user-2/);
  await expect(page.getByText('Private direct conversation')).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText('Start with a simple hello')).toBeVisible();
  await expect(page.getByText(/Golden Gate Park Trail/i)).toHaveCount(0);
  await expect(page.getByText(/Sarah Johnson/i)).toHaveCount(0);
});
