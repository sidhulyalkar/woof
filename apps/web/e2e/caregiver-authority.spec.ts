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

const browserUser = {
  id: 'caregiver-user',
  email: 'caregiver@example.com',
  handle: 'trusted-caregiver',
  pets: [],
};

async function authenticate(page: Page) {
  const token = 'caregiver-browser-token';
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

const companionState = {
  mode: 'FOSTER_CAREGIVER',
  modeSource: 'PERSISTED',
  hasAuthorizedPet: false,
  landing: 'COMPANION_TODAY',
  authority: {
    modeControlsPresentation: true,
    petAccessComesFromRelationships: true,
    modeNeverCreatesPetAuthority: true,
  },
};

const pet = {
  id: 'pet-1',
  name: 'Nova',
  species: 'DOG',
  breed: 'Husky mix',
  avatarUrl: null,
};

function grant(status: 'PENDING_ACCEPTANCE' | 'ACTIVE', expiresAt: string) {
  return {
    id: 'grant-1',
    petId: pet.id,
    issuerUserId: 'owner-1',
    recipientUserId: browserUser.id,
    requestKey: 'caregiver-browser-request-1',
    policyVersion: 'caregiver-authority-v1',
    status,
    effectiveStatus: status,
    issuedAt: '2026-08-29T07:00:00.000Z',
    acceptedAt: status === 'ACTIVE' ? '2026-08-29T07:01:00.000Z' : null,
    declinedAt: null,
    expiresAt,
    revokedAt: null,
    revokedByUserId: null,
    createdAt: '2026-08-29T07:00:00.000Z',
    updatedAt: '2026-08-29T07:01:00.000Z',
    capabilities: ['LOG_OBSERVATION', 'VIEW_TODAY'],
    pet,
    issuerHandle: 'nova-guardian',
    relationshipBlocked: false,
  };
}

function caregiverToday(expiresAt: string) {
  return {
    pet,
    relationship: {
      grantId: 'grant-1',
      issuerUserId: 'owner-1',
      issuerHandle: 'nova-guardian',
      expiresAt,
      capabilities: ['LOG_OBSERVATION', 'VIEW_TODAY'],
      effectiveStatus: 'ACTIVE',
    },
    available: { viewToday: true, logObservation: true },
    boundaries: {
      householdHistory: false,
      siblingPets: false,
      medicalAuthority: false,
      profileCorrection: false,
      connectorAdmin: false,
      bondXpAuthority: false,
      recommendationEvidenceAuthority: false,
    },
  };
}

async function mockCompanionShell(page: Page) {
  await page.route('**/companion/state', (route) => fulfillJson(route, companionState));
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
}

test.describe('dogOS Caregiver Authority', () => {
  test('invitation -> accept -> caregiver Today -> context observation -> revoke fails closed', async ({
    page,
  }) => {
    await authenticate(page);
    await mockCompanionShell(page);

    let accepted = false;
    let revoked = false;
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000).toISOString();

    await page.route('**/caregiver/grants/received', (route) =>
      fulfillJson(route, [grant(accepted ? 'ACTIVE' : 'PENDING_ACCEPTANCE', expiresAt)])
    );
    await page.route('**/caregiver/pets', (route) =>
      fulfillJson(route, accepted && !revoked ? [grant('ACTIVE', expiresAt)] : [])
    );
    await page.route('**/caregiver/grants/grant-1/accept', async (route) => {
      accepted = true;
      await fulfillJson(route, { ...grant('ACTIVE', expiresAt), replayed: false });
    });
    await page.route('**/caregiver/pets/pet-1/today', (route) =>
      revoked
        ? fulfillJson(route, { message: 'Caregiver pet access not found' }, 404)
        : fulfillJson(route, caregiverToday(expiresAt))
    );
    await page.route('**/caregiver/pets/pet-1/observations', (route) =>
      revoked
        ? fulfillJson(route, { message: 'Caregiver pet access not found' }, 404)
        : fulfillJson(route, {
            id: 'observation-1',
            grantId: 'grant-1',
            petId: pet.id,
            actorUserId: browserUser.id,
            authorityClass: 'CONTEXT_ONLY',
            kind: 'ROUTINE',
            summary: 'Settled after the evening routine.',
            note: null,
            context: { policyVersion: 'caregiver-authority-v1' },
            observedAt: new Date().toISOString(),
            createdAt: new Date().toISOString(),
          })
    );

    await page.goto('/');

    await expect(page.locator('[data-caregiver-pending-grant="grant-1"]')).toBeVisible({
      timeout: 10_000,
    });
    await page.getByRole('button', { name: 'Accept temporary care' }).click();
    await expect(page.locator('[data-caregiver-active-grant="grant-1"]')).toBeVisible();
    await page.getByRole('link', { name: 'Open caregiver Today' }).click();

    await expect(page.locator('[data-caregiver-today]')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Care for Nova today.' })).toBeVisible();
    await expect(page.getByText('Household history')).toBeVisible();
    await expect(page.getByText('Not granted').first()).toBeVisible();
    await expect(page.getByRole('link', { name: 'Compass', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Story', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Auto', exact: true })).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Coach', exact: true })).toHaveCount(0);

    await page.locator('[data-caregiver-observation-summary]').fill('Settled after the evening routine.');
    await page.locator('[data-caregiver-submit-observation]').click();
    await expect(page.locator('[data-caregiver-observation-saved]')).toContainText('context only');

    revoked = true;
    await page.reload();

    await expect(
      page.getByRole('heading', { name: 'Caregiver access is not available' })
    ).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('[data-caregiver-observation-summary]')).toHaveCount(0);
    await expect(page.locator('[data-caregiver-today]')).toHaveCount(0);
  });

  test('caregiver Today becomes unavailable at the local expiry boundary', async ({ page }) => {
    await authenticate(page);
    const expiresAt = new Date(Date.now() + 900).toISOString();

    await page.route('**/caregiver/pets/pet-1/today', (route) =>
      fulfillJson(route, caregiverToday(expiresAt))
    );

    await page.goto('/caregiver/pets/pet-1');
    await expect(page.locator('[data-caregiver-today]')).toBeVisible({ timeout: 10_000 });
    await expect(
      page.getByRole('heading', { name: 'Caregiver access is not available' })
    ).toBeVisible({ timeout: 5_000 });
    await expect(page.locator('[data-caregiver-observation-summary]')).toHaveCount(0);
  });
});
