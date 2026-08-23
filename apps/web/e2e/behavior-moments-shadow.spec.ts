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

async function seedSession(page: Page) {
  await page.addInitScript(() => {
    const pet = {
      id: 'pet-1',
      name: 'Nova',
      species: 'DOG',
      breed: 'Husky mix',
    };
    const user = {
      id: 'user-1',
      email: 'owner@example.com',
      handle: 'nova-human',
      pets: [pet],
    };
    window.localStorage.setItem('authToken', 'shadow-browser-token');
    window.localStorage.setItem(
      'woof-session-storage',
      JSON.stringify({
        state: {
          user,
          pets: [pet],
          token: 'shadow-browser-token',
          refreshToken: null,
          isAuthenticated: true,
        },
        version: 0,
      })
    );
  });
}

test('Shadow Lab renders reviewable evidence without requesting compatibility authority', async ({
  page,
}) => {
  await seedSession(page);
  let compatibilityRequests = 0;

  page.on('request', (request) => {
    if (request.url().includes('/compatibility/')) compatibilityRequests += 1;
  });

  await page.route('**/behavior-vision/shadow**', (route) =>
    fulfillJson(route, {
      policy: {
        version: 'woof-behavior-shadow-v1',
        mode: 'shadow-evidence-only',
        canInfluenceCompatibility: false,
        canMutateCanonicalPetState: false,
        canMakeSafetyDecision: false,
        promotionEnabled: false,
        promotionRequiresSeparateQualifiedRelease: true,
      },
      evaluation: {
        observations: 24,
        usableObservations: 20,
        ownerReviewedObservations: 12,
        ownerConfirmedObservations: 11,
        ownerRejectedObservations: 1,
        ownerUnreviewedObservations: 8,
        confirmationRate: 11 / 12,
        usableRate: 20 / 24,
        contextsSeen: 4,
        pairedSessions: 6,
        personalizationConfidence: 0.72,
        modelVersions: ['behavior-shadow-model-1'],
        evidenceReady: true,
        readinessGates: {
          usableObservations: 20,
          ownerReviewedObservations: 10,
          confirmationRate: 0.8,
          contexts: 3,
          pairedSessions: 5,
        },
      },
      moments: [
        {
          observationId: 'obs-1',
          observationCreatedAt: '2026-08-23T20:00:00.000Z',
          context: 'park',
          phase: 'recovery',
          startMs: 4200,
          endMs: 6800,
          confidence: 0.91,
          labels: ['oriented toward dog', 'forward movement'],
          sources: ['pose', 'motion'],
        },
      ],
    })
  );

  await page.goto('/coach/observe/shadow');

  await expect(page.getByRole('heading', { name: 'Shadow Lab' })).toBeVisible();
  await expect(page.getByText('zero authority')).toBeVisible();
  await expect(page.getByText('Evidence gates met')).toBeVisible();
  await expect(page.getByText('0:04.2–0:06.8')).toBeVisible();
  await expect(page.getByText('oriented toward dog · forward movement')).toBeVisible();
  await expect(page.getByText(/Compatibility influence: off/i)).toBeVisible();
  expect(compatibilityRequests).toBe(0);
});
