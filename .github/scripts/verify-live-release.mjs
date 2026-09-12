#!/usr/bin/env node

import process from 'node:process';

const SHA_PATTERN = /^[0-9a-f]{40}$/;
const INVALID_ORIGIN = 'https://woof-invalid-origin.example';

function requireValue(name, value) {
  if (!value || !value.trim()) throw new Error(`${name} is required`);
  return value.trim();
}

function requireHttpsOrigin(name, value) {
  const raw = requireValue(name, value);
  const parsed = new URL(raw);
  if (parsed.protocol !== 'https:' || parsed.origin !== raw.replace(/\/$/, '')) {
    throw new Error(`${name} must be an exact HTTPS origin without a path, query, or fragment`);
  }
  return parsed.origin;
}

function requireApiUrl(value) {
  const raw = requireValue('API_URL', value).replace(/\/$/, '');
  const parsed = new URL(raw);
  if (parsed.protocol !== 'https:' || parsed.username || parsed.password) {
    throw new Error('API_URL must be an HTTPS URL without credentials');
  }
  return raw;
}

function assertReleaseBody(label, body, expectedRelease) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new Error(`${label} did not return a JSON object`);
  }
  if (body.release !== expectedRelease) {
    throw new Error(`${label} release mismatch: expected ${expectedRelease}, received ${body.release}`);
  }
}

function assertSecurityHeaders(label, response) {
  if (response.headers.get('x-content-type-options') !== 'nosniff') {
    throw new Error(`${label} is missing X-Content-Type-Options: nosniff`);
  }
}

async function fetchWithTimeout(url, options = {}) {
  return fetch(url, {
    ...options,
    redirect: 'follow',
    signal: AbortSignal.timeout(15_000),
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetchWithTimeout(url, options);
  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      throw new Error(`${url} returned non-JSON content`);
    }
  }
  return { response, body };
}

async function verifyLiveRelease({ apiUrl, webOrigin, expectedRelease }) {
  for (const endpoint of ['live', 'ready']) {
    const url = `${apiUrl}/ops/health/${endpoint}`;
    const { response, body } = await fetchJson(url, {
      headers: { Origin: webOrigin },
    });
    if (response.status !== 200) {
      throw new Error(`${endpoint} returned HTTP ${response.status}`);
    }
    assertReleaseBody(endpoint, body, expectedRelease);
    assertSecurityHeaders(endpoint, response);
    if (response.headers.get('access-control-allow-origin') !== webOrigin) {
      throw new Error(`${endpoint} did not authorize the configured Web origin`);
    }
    if (response.headers.get('access-control-allow-credentials') !== 'true') {
      throw new Error(`${endpoint} did not expose credentialed CORS authority`);
    }
  }

  const authResponse = await fetchWithTimeout(`${apiUrl}/auth/me`, {
    headers: { Origin: webOrigin },
  });
  if (authResponse.status !== 401) {
    throw new Error(`unauthenticated /auth/me must fail closed with 401, received ${authResponse.status}`);
  }
  assertSecurityHeaders('auth/me', authResponse);

  for (const docsUrl of [new URL('/docs', apiUrl).toString(), `${apiUrl}/docs`]) {
    const response = await fetchWithTimeout(docsUrl);
    if (response.ok) {
      throw new Error(`production-shaped API documentation is unexpectedly reachable at ${docsUrl}`);
    }
  }

  // Repository qualification already proves hostile origins are rejected. Avoid
  // deliberately generating a production 5xx solely to prove that negative path.
  if (webOrigin === INVALID_ORIGIN) {
    throw new Error('configured Web origin must not equal the reserved invalid-origin probe value');
  }

  console.log(`live release smoke verified for ${expectedRelease} via ${webOrigin}`);
}

function selfTest() {
  const sha = 'a'.repeat(40);
  assertReleaseBody('self-test', { release: sha }, sha);

  let rejected = false;
  try {
    assertReleaseBody('self-test', { release: 'b'.repeat(40) }, sha);
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error('release mismatch self-test failed');

  if (requireHttpsOrigin('WEB_ORIGIN', 'https://woof.example') !== 'https://woof.example') {
    throw new Error('HTTPS origin self-test failed');
  }

  rejected = false;
  try {
    requireHttpsOrigin('WEB_ORIGIN', 'https://woof.example/path');
  } catch {
    rejected = true;
  }
  if (!rejected) throw new Error('path-bearing Web origin self-test failed');

  console.log('live release smoke verifier self-test passed');
}

if (process.argv.includes('--self-test')) {
  selfTest();
} else {
  const apiUrl = requireApiUrl(process.env.API_URL);
  const webOrigin = requireHttpsOrigin('WEB_ORIGIN', process.env.WEB_ORIGIN);
  const expectedRelease = requireValue('EXPECTED_RELEASE', process.env.EXPECTED_RELEASE);
  if (!SHA_PATTERN.test(expectedRelease)) {
    throw new Error('EXPECTED_RELEASE must be an exact lowercase 40-character Git SHA');
  }
  await verifyLiveRelease({ apiUrl, webOrigin, expectedRelease });
}
