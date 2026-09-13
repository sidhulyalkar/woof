import { createServer, type Server } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { test, expect } from '@playwright/test';

const LEGACY_WORKER = `
const CACHE_NAME = 'petpath-v1';

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_NAME);
    await cache.put(
      '/api/v1/auth/me',
      new Response(JSON.stringify({ source: 'legacy-cache', stale: true }), {
        headers: { 'Content-Type': 'application/json' },
      })
    );
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  if (new URL(event.request.url).pathname === '/api/v1/auth/me') {
    event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
  }
});
`;

let server: Server;
let origin: string;
let cleanupWorker: string;

test.describe.configure({ mode: 'serial' });

test.beforeAll(async () => {
  cleanupWorker = await readFile(resolve(process.cwd(), 'public/sw.js'), 'utf8');

  server = createServer((request, response) => {
    const path = new URL(request.url ?? '/', 'http://127.0.0.1').pathname;

    if (path === '/legacy-sw.js') {
      response.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'Cache-Control': 'no-store',
        'Service-Worker-Allowed': '/',
      });
      response.end(LEGACY_WORKER);
      return;
    }

    if (path === '/sw.js') {
      response.writeHead(200, {
        'Content-Type': 'application/javascript; charset=utf-8',
        'Cache-Control': 'no-store',
        'Service-Worker-Allowed': '/',
      });
      response.end(cleanupWorker);
      return;
    }

    if (path === '/api/v1/auth/me') {
      response.writeHead(200, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      response.end(JSON.stringify({ source: 'network', stale: false }));
      return;
    }

    response.writeHead(200, {
      'Content-Type': 'text/html; charset=utf-8',
      'Cache-Control': 'no-store',
    });
    response.end('<!doctype html><title>Woof worker retirement fixture</title>');
  });

  await new Promise<void>((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolveListen());
  });

  const address = server.address();
  if (!address || typeof address === 'string') {
    throw new Error('Legacy worker fixture did not bind a TCP port');
  }
  origin = `http://127.0.0.1:${address.port}`;
});

test.afterAll(async () => {
  if (!server) return;
  await new Promise<void>((resolveClose, reject) =>
    server.close((error) => (error ? reject(error) : resolveClose()))
  );
});

test('retires a legacy cache-first registration without serving stale protected state afterward', async ({
  page,
}) => {
  await page.goto(origin);

  await page.evaluate(async () => {
    const registration = await navigator.serviceWorker.register('/legacy-sw.js', { scope: '/' });
    await navigator.serviceWorker.ready;
    if (!registration.active) {
      await new Promise<void>((resolveActive) => {
        const worker = registration.installing ?? registration.waiting;
        if (!worker) return resolveActive();
        worker.addEventListener('statechange', () => {
          if (worker.state === 'activated') resolveActive();
        });
      });
    }
  });

  // Reload once so the legacy worker unquestionably controls the document.
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)))
    .toBe(true);

  const staleResponse = await page.evaluate(async () => {
    const response = await fetch('/api/v1/auth/me');
    return response.json();
  });
  expect(staleResponse).toEqual({ source: 'legacy-cache', stale: true });

  await page.evaluate(async () => {
    const extra = await caches.open('petpath-historical-private');
    await extra.put('/private-history', new Response('legacy-private-state'));
    await navigator.serviceWorker.register('/sw.js', { scope: '/' });
  });

  await expect
    .poll(
      () =>
        page.evaluate(async () => ({
          registration: Boolean(await navigator.serviceWorker.getRegistration('/')),
          legacyCaches: (await caches.keys()).filter((name) => name.startsWith('petpath-')),
        })),
      { timeout: 10_000 }
    )
    .toEqual({ registration: false, legacyCaches: [] });

  // An unregistered worker may continue controlling the current document until
  // unload. A fresh navigation is the historical-client boundary that matters.
  await page.reload();
  await expect
    .poll(() => page.evaluate(() => Boolean(navigator.serviceWorker.controller)))
    .toBe(false);

  const networkResponse = await page.evaluate(async () => {
    const response = await fetch('/api/v1/auth/me');
    return response.json();
  });
  expect(networkResponse).toEqual({ source: 'network', stale: false });
});
