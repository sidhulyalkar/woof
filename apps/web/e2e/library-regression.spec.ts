import AxeBuilder from '@axe-core/playwright';
import { seedAuthenticatedSession } from './support/session';
import { expect, test, type Page, type Route } from '@playwright/test';

const user = {
  id: 'user-1',
  email: 'beta@example.com',
  handle: 'beta-owner',
  pets: [{ id: 'pet-1', name: 'Nova', species: 'DOG', breed: 'Mixed' }],
};

const tinyImage =
  'data:image/svg+xml;charset=utf-8,' +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><rect width="400" height="400" fill="#ece8f8"/><circle cx="200" cy="190" r="80" fill="#8b5cf6" opacity=".22"/></svg>'
  );

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

async function fulfillPreflight(route: Route) {
  if (route.request().method() !== 'OPTIONS') return false;
  await route.fulfill({ status: 204, headers: corsHeaders(route), body: '' });
  return true;
}

async function fulfillJson(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    headers: {
      ...corsHeaders(route),
      'content-type': 'application/json',
    },
    body: JSON.stringify(body),
  });
}

function asset(id: string, favorite = false) {
  return {
    id,
    createdAt: '2026-08-21T07:00:00.000Z',
    filename: `${id}.jpg`,
    mimeType: 'image/jpeg',
    mediaType: 'image',
    sizeBytes: 1024,
    capturedAt: '2026-08-21T06:00:00.000Z',
    source: 'device-picker',
    provider: 'DEVICE',
    favorite,
    albumIds: id === 'asset-1' ? ['album-1'] : [],
    smartAlbumIds: favorite ? ['smart:favorites'] : ['smart:recent'],
    tags: [{ label: 'imported', source: 'system' }],
    linkedObservationIds: [],
    url: tinyImage,
    thumbnailUrl: tinyImage,
    posterUrl: null,
    previewUrl: null,
    urlExpiresInSeconds: 900,
    status: 'READY',
  };
}

function libraryPayload(assets: ReturnType<typeof asset>[] = []) {
  return {
    petId: 'pet-1',
    assets,
    albums: [
      {
        id: 'smart:recent',
        name: 'Recent',
        description: 'Recent',
        icon: 'clock',
        kind: 'SMART',
        count: assets.length,
      },
      {
        id: 'smart:favorites',
        name: 'Favorites',
        description: 'Favorites',
        icon: 'heart',
        kind: 'SMART',
        count: assets.filter((item) => item.favorite).length,
      },
      {
        id: 'album-1',
        name: 'Weekend adventures',
        description: 'Trail days',
        icon: 'folder',
        kind: 'USER',
        count: assets.filter((item) => item.albumIds.includes('album-1')).length,
      },
    ],
    storage: {
      usedBytes: assets.length * 1024,
      quotaBytes: 10 * 1024 * 1024 * 1024,
      storageConfigured: true,
    },
    importCapabilities: {
      devicePicker: true,
      appleSystemPicker: true,
      googlePhotosPicker: true,
      googlePhotosBroadLibrarySync: false,
    },
  };
}

async function authenticate(page: Page) {
  await seedAuthenticatedSession(page, {
    user,
    token: 'browser-test-token',
  });
}

async function routeLibrary(page: Page, handler: (route: Route) => Promise<void>) {
  await page.route('**/media-library**', async (route) => {
    if (await fulfillPreflight(route)) return;
    if (route.request().method() === 'GET') return handler(route);
    return route.fallback();
  });
}

test.describe('Private pet media library', () => {
  test('empty library is calm, keyboard reachable, and free of serious WCAG violations', async ({
    page,
  }) => {
    await authenticate(page);
    await routeLibrary(page, async (route) => {
      await fulfillJson(route, libraryPayload());
    });
    await page.goto('/library');

    await expect(page.getByTestId('media-library-empty')).toBeVisible();
    await expect(page.getByRole('heading', { name: /moments that teach woof/i })).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(
      results.violations.filter((violation) =>
        ['serious', 'critical'].includes(violation.impact ?? '')
      )
    ).toEqual([]);

    await page.keyboard.press('Tab');
    expect(await page.evaluate(() => document.activeElement?.tagName)).not.toBe('BODY');
  });

  test('shows an actionable failure state rather than an empty library', async ({ page }) => {
    await authenticate(page);
    await routeLibrary(page, async (route) => {
      await fulfillJson(route, { message: 'storage unavailable' }, 503);
    });
    await page.goto('/library');
    await expect(page.getByTestId('media-library-error')).toBeVisible();
    await expect(page.getByRole('button', { name: /try again/i })).toBeVisible();
  });

  test('announces upload progress while a private direct upload is in flight', async ({ page }) => {
    await authenticate(page);
    await routeLibrary(page, async (route) => {
      await fulfillJson(route, libraryPayload());
    });
    await page.route('**/media-library/uploads/intents', async (route) => {
      if (await fulfillPreflight(route)) return;
      await fulfillJson(route, {
        assetId: 'asset-new',
        uploadUrl: 'https://upload.woof.test/private',
        requiredHeaders: { 'Content-Type': 'image/jpeg' },
      });
    });
    await page.route('https://upload.woof.test/private', async (route) => {
      if (await fulfillPreflight(route)) return;
      await new Promise((resolve) => setTimeout(resolve, 1200));
      await route.fulfill({ status: 200, headers: corsHeaders(route), body: '' });
    });
    await page.route('**/media-library/uploads/complete', async (route) => {
      if (await fulfillPreflight(route)) return;
      await fulfillJson(route, asset('asset-new'));
    });

    await page.goto('/library');
    await page.locator('input[type="file"]').setInputFiles({
      name: 'nova.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x01]),
    });
    await expect(page.getByTestId('media-library-uploading')).toBeVisible();
  });

  test('album filtering and multi-selection keep hierarchy stable', async ({ page }) => {
    await authenticate(page);
    const assets = [asset('asset-1'), asset('asset-2', true)];
    await routeLibrary(page, async (route) => {
      const url = new URL(route.request().url());
      const filtered = url.searchParams.get('albumId') === 'album-1' ? [assets[0]] : assets;
      await fulfillJson(route, libraryPayload(filtered));
    });
    await page.goto('/library');

    const headingBox = await page
      .getByRole('heading', { name: /moments that teach woof/i })
      .boundingBox();
    const albumsBox = await page.getByTestId('media-library-albums').boundingBox();
    expect(headingBox && albumsBox && headingBox.y < albumsBox.y).toBeTruthy();

    await page.getByRole('button', { name: /weekend adventures/i }).click();
    await expect(page.getByTestId('media-library-grid').locator('[data-asset-id]')).toHaveCount(1);

    await page.getByRole('button', { name: /select asset-1[.]jpg/i }).click();
    await expect(page.getByText(/1 selected/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /export 1 to google/i })).toBeVisible();
  });

  test('mobile library has no horizontal overflow in populated state', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await authenticate(page);
    await routeLibrary(page, async (route) => {
      await fulfillJson(route, libraryPayload([asset('asset-1'), asset('asset-2')]));
    });
    await page.goto('/library');
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth
    );
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.getByTestId('media-library-grid')).toBeVisible();
  });
});
