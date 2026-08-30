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

const pet = {
  id: 'pet-1',
  name: 'Nova',
  species: 'DOG',
  breed: 'Husky mix',
};

const browserUser = {
  id: 'user-1',
  email: 'owner@example.com',
  handle: 'nova-human',
  pets: [pet],
};

async function authenticate(page: Page) {
  const token = 'shadow-browser-token';
  await page.route('**/auth/me', (route) => fulfillJson(route, browserUser));
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
    { token, user: browserUser }
  );
}

test('Shadow Lab renders active-release evidence without requesting compatibility authority', async ({
  page,
}) => {
  await authenticate(page);
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
        requiresQualifiedModelRelease: true,
        learningScope: 'active-qualified-release-only',
      },
      evaluation: {
        observations: 24,
        qualifiedObservations: 22,
        activeReleaseObservations: 20,
        inactiveQualifiedObservations: 2,
        unqualifiedObservations: 2,
        usableObservations: 20,
        ownerReviewedObservations: 12,
        ownerConfirmedObservations: 11,
        ownerRejectedObservations: 1,
        ownerUnreviewedObservations: 8,
        confirmationRate: 11 / 12,
        usableRate: 1,
        contextsSeen: 4,
        pairedSessions: 6,
        personalizationConfidence: 0.72,
        activeReleaseId: 'behavior-shadow-2026-08-27',
        qualifiedReleaseIds: ['behavior-shadow-2026-07-01', 'behavior-shadow-2026-08-27'],
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
  await expect(page.getByText('Active qualified release only')).toBeVisible();
  await expect(
    page.getByText(/2 older qualified and 2 legacy unqualified observations/i)
  ).toBeVisible();
  await expect(page.getByText(/Active release: behavior-shadow-2026-08-27/)).toBeVisible();
  await expect(page.getByText('Evidence gates met')).toBeVisible();
  await expect(page.getByText('0:04.2–0:06.8')).toBeVisible();
  await expect(page.getByText('oriented toward dog · forward movement')).toBeVisible();
  await expect(page.getByText(/Compatibility influence: off/i)).toBeVisible();
  expect(compatibilityRequests).toBe(0);
});
