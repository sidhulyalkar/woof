#!/usr/bin/env node

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { verifyWebDeploymentProvenance } from './verify-web-deployment-provenance.mjs';

const GIT_SHA_PATTERN = /^[0-9a-f]{40}$/;
const ALLOWED_ENVIRONMENTS = new Set(['staging', 'production']);

function exactRelease(value) {
  const normalized = value.trim().toLowerCase();
  if (!GIT_SHA_PATTERN.test(normalized)) {
    throw new Error(`Expected release must be one exact 40-hex Git SHA, received '${value}'.`);
  }
  return normalized;
}

function httpsUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${label} is invalid: '${value}'.`);
  }

  if (parsed.protocol !== 'https:') {
    throw new Error(`${label} must use HTTPS, received '${value}'.`);
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${label} must not contain credentials, query parameters, or fragments.`);
  }
  return parsed;
}

function apiEndpoint(apiUrl, suffix) {
  const endpoint = new URL(apiUrl.toString());
  endpoint.pathname = `${endpoint.pathname.replace(/\/$/, '')}${suffix}`;
  return endpoint.toString();
}

async function sleep(ms) {
  if (ms > 0) await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(fetchImpl, url, options, attempts, delayMs) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetchImpl(url, options);
      if (response.ok) return response;
      lastError = new Error(`${url} returned HTTP ${response.status}.`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
    }

    if (attempt < attempts) await sleep(delayMs);
  }
  throw lastError ?? new Error(`Unable to fetch ${url}.`);
}

async function jsonResponse(response, label) {
  try {
    return await response.json();
  } catch {
    throw new Error(`${label} did not return valid JSON.`);
  }
}

function verifyCors(response, expectedWebOrigin) {
  const allowOrigin = response.headers.get('access-control-allow-origin');
  const allowCredentials = response.headers.get('access-control-allow-credentials');

  if (allowOrigin !== expectedWebOrigin) {
    throw new Error(
      `Production CORS mismatch: expected '${expectedWebOrigin}', received '${allowOrigin}'.`
    );
  }
  if (allowCredentials !== 'true') {
    throw new Error(
      `Production CORS credentials mismatch: expected 'true', received '${allowCredentials}'.`
    );
  }

  return { allowOrigin, allowCredentials: true };
}

async function verifyWebSurface({
  fetchImpl,
  url,
  expectedRelease,
  expectedApiUrl,
  attempts,
  delayMs,
  label,
}) {
  const response = await fetchWithRetry(
    fetchImpl,
    url,
    { headers: { Accept: 'text/html', 'Cache-Control': 'no-store' } },
    attempts,
    delayMs
  );
  const html = await response.text();
  try {
    return verifyWebDeploymentProvenance({
      html,
      expectedRelease,
      expectedApiUrl,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`${label} provenance failed: ${message}`);
  }
}

export async function qualifyLiveRelease({
  environment,
  expectedRelease,
  expectedApiUrl,
  webDeploymentUrl,
  expectedWebOrigin,
  fetchImpl = globalThis.fetch,
  attempts = 8,
  delayMs = 3000,
  now = () => new Date(),
}) {
  if (!ALLOWED_ENVIRONMENTS.has(environment)) {
    throw new Error(
      `Environment must be one of ${[...ALLOWED_ENVIRONMENTS].join(', ')}, received '${environment}'.`
    );
  }
  if (typeof fetchImpl !== 'function') {
    throw new Error('A fetch implementation is required.');
  }
  if (!Number.isInteger(attempts) || attempts < 1 || attempts > 20) {
    throw new Error(`Attempts must be an integer from 1 to 20, received '${attempts}'.`);
  }
  if (!Number.isInteger(delayMs) || delayMs < 0 || delayMs > 30000) {
    throw new Error(`Retry delay must be an integer from 0 to 30000 ms, received '${delayMs}'.`);
  }

  const release = exactRelease(expectedRelease);
  const apiUrl = httpsUrl(expectedApiUrl, 'Expected API URL');
  const deploymentUrl = httpsUrl(webDeploymentUrl, 'Web deployment URL');
  const publicWebUrl = httpsUrl(expectedWebOrigin, 'Expected Web origin');
  if (
    publicWebUrl.pathname !== '/' ||
    publicWebUrl.origin !== publicWebUrl.toString().replace(/\/$/, '')
  ) {
    throw new Error(`Expected Web origin must be an origin only, received '${expectedWebOrigin}'.`);
  }

  const webOrigin = publicWebUrl.origin;
  const requestOptions = {
    headers: {
      Accept: 'application/json',
      Origin: webOrigin,
      'Cache-Control': 'no-store',
    },
  };

  const liveResponse = await fetchWithRetry(
    fetchImpl,
    apiEndpoint(apiUrl, '/ops/health/live'),
    requestOptions,
    attempts,
    delayMs
  );
  const cors = verifyCors(liveResponse, webOrigin);
  const live = await jsonResponse(liveResponse, 'API liveness');
  if (live.status !== 'live' || live.release !== release) {
    throw new Error(
      `API liveness mismatch: expected status=live release=${release}, received status=${live.status} release=${live.release}.`
    );
  }

  const readyResponse = await fetchWithRetry(
    fetchImpl,
    apiEndpoint(apiUrl, '/ops/health/ready'),
    requestOptions,
    attempts,
    delayMs
  );
  const ready = await jsonResponse(readyResponse, 'API readiness');
  if (
    ready.status !== 'ready' ||
    ready.release !== release ||
    ready.database?.status !== 'ready'
  ) {
    throw new Error(
      `API readiness mismatch: expected ready database and release ${release}, received status=${ready.status} database=${ready.database?.status} release=${ready.release}.`
    );
  }

  const deploymentDemoUrl = new URL('/demo', deploymentUrl.origin).toString();
  const publicDemoUrl = new URL('/demo', webOrigin).toString();

  const deploymentWeb = await verifyWebSurface({
    fetchImpl,
    url: deploymentDemoUrl,
    expectedRelease: release,
    expectedApiUrl,
    attempts,
    delayMs,
    label: 'Web deployment URL',
  });

  const publicWeb =
    deploymentUrl.origin === webOrigin
      ? deploymentWeb
      : await verifyWebSurface({
          fetchImpl,
          url: publicDemoUrl,
          expectedRelease: release,
          expectedApiUrl,
          attempts,
          delayMs,
          label: 'Public Web origin',
        });

  return {
    schemaVersion: 1,
    kind: 'woof-live-release-qualification',
    environment,
    qualifiedAt: now().toISOString(),
    releaseSha: release,
    api: {
      baseUrl: expectedApiUrl,
      liveStatus: live.status,
      readyStatus: ready.status,
      databaseStatus: ready.database.status,
    },
    web: {
      deploymentUrl: deploymentUrl.origin,
      publicOrigin: webOrigin,
      route: '/demo',
      deploymentRelease: deploymentWeb.release,
      publicRelease: publicWeb.release,
      apiOrigin: publicWeb.apiOrigin,
    },
    cors,
    checks: [
      'api-liveness',
      'api-readiness',
      'api-release-identity',
      'web-deployment-provenance',
      'web-public-origin-provenance',
      'web-api-origin',
      'cors-public-origin',
      'cors-credentials',
    ],
  };
}

function response(body, { status = 200, headers = {}, contentType = 'application/json' } = {}) {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status,
    headers: { 'content-type': contentType, ...headers },
  });
}

async function selfTest() {
  const release = '0123456789abcdef0123456789abcdef01234567';
  const api = 'https://api.example.com/api/v1';
  const webDeployment = 'https://deployment.example.com';
  const webOrigin = 'https://www.example.com';
  const corsHeaders = {
    'access-control-allow-origin': webOrigin,
    'access-control-allow-credentials': 'true',
  };
  const goodHtml = `<html><head><meta name="woof-release" content="${release}"><meta name="woof-api-origin" content="${api}"></head></html>`;

  const buildFetch = ({
    liveRelease = release,
    readyStatus = 'ready',
    allowOrigin = webOrigin,
    publicRelease = release,
  } = {}) =>
    async (url) => {
      if (url.endsWith('/ops/health/live')) {
        return response(
          { status: 'live', release: liveRelease },
          { headers: { ...corsHeaders, 'access-control-allow-origin': allowOrigin } }
        );
      }
      if (url.endsWith('/ops/health/ready')) {
        return response({
          status: readyStatus,
          release,
          database: { status: readyStatus === 'ready' ? 'ready' : 'unavailable' },
        });
      }
      if (url === `${webDeployment}/demo`) {
        return response(goodHtml, { contentType: 'text/html' });
      }
      if (url === `${webOrigin}/demo`) {
        return response(goodHtml.replace(release, publicRelease), { contentType: 'text/html' });
      }
      return response({ error: 'not found' }, { status: 404 });
    };

  const input = {
    environment: 'staging',
    expectedRelease: release,
    expectedApiUrl: api,
    webDeploymentUrl: webDeployment,
    expectedWebOrigin: webOrigin,
    attempts: 1,
    delayMs: 0,
    now: () => new Date('2026-09-11T00:00:00.000Z'),
  };

  const receipt = await qualifyLiveRelease({ ...input, fetchImpl: buildFetch() });
  if (
    receipt.releaseSha !== release ||
    receipt.api.databaseStatus !== 'ready' ||
    receipt.web.publicRelease !== release
  ) {
    throw new Error('Self-test failed to produce the expected qualification receipt.');
  }

  for (const [label, fetchImpl] of [
    ['wrong API release', buildFetch({ liveRelease: 'f'.repeat(40) })],
    ['database not ready', buildFetch({ readyStatus: 'not_ready' })],
    ['wrong CORS origin', buildFetch({ allowOrigin: 'https://wrong.example.com' })],
    ['stale public Web alias', buildFetch({ publicRelease: 'f'.repeat(40) })],
  ]) {
    let rejected = false;
    try {
      await qualifyLiveRelease({ ...input, fetchImpl });
    } catch {
      rejected = true;
    }
    if (!rejected) throw new Error(`Self-test failed to reject ${label}.`);
  }

  let rejectedInsecure = false;
  try {
    await qualifyLiveRelease({
      ...input,
      expectedApiUrl: 'http://api.example.com/api/v1',
      fetchImpl: buildFetch(),
    });
  } catch {
    rejectedInsecure = true;
  }
  if (!rejectedInsecure) throw new Error('Self-test failed to reject insecure API URL.');

  console.log('Live release qualification self-test passed.');
}

async function main() {
  if (process.argv.includes('--self-test')) {
    await selfTest();
    return;
  }

  const receipt = await qualifyLiveRelease({
    environment: process.env.WOOF_ENVIRONMENT ?? '',
    expectedRelease: process.env.EXPECTED_RELEASE ?? '',
    expectedApiUrl: process.env.EXPECTED_API_URL ?? '',
    webDeploymentUrl: process.env.WEB_URL ?? '',
    expectedWebOrigin: process.env.EXPECTED_WEB_ORIGIN ?? '',
  });

  const receiptPath = process.env.RELEASE_RECEIPT_PATH;
  if (receiptPath) {
    await mkdir(dirname(receiptPath), { recursive: true });
    await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf8');
    console.log(`Wrote privacy-safe live release receipt to ${receiptPath}.`);
  }

  console.log(
    `Qualified ${receipt.environment} release ${receipt.releaseSha} at ${receipt.web.publicOrigin}.`
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
