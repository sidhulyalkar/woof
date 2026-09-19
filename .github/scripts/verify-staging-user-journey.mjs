#!/usr/bin/env node

import { randomBytes, randomUUID } from 'node:crypto';
import { pathToFileURL } from 'node:url';

const DEFAULT_TIMEOUT_MS = 15_000;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function normalizeApiUrl(value) {
  const url = new URL(value);
  if (url.protocol !== 'https:' && url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') {
    throw new Error('staging journey requires HTTPS outside local self-test');
  }
  return url.href.replace(/\/$/, '');
}

function createSyntheticIdentity(uuid = randomUUID()) {
  const compact = uuid.replaceAll('-', '').slice(0, 16);
  return {
    email: `woof.blackbox+${compact}@example.com`,
    handle: `bb_${compact}`.slice(0, 30),
    password: `Wf!${randomBytes(18).toString('base64url')}`,
    registrationKey: uuid,
    petCreationKey: `staging-journey:${uuid}`,
  };
}

async function requestJson(fetchImpl, apiUrl, path, options = {}) {
  const response = await fetchImpl(`${apiUrl}${path}`, {
    ...options,
    headers: {
      accept: 'application/json',
      ...(options.body ? { 'content-type': 'application/json' } : {}),
      ...(options.headers ?? {}),
    },
    signal: options.signal ?? AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
  });

  let body = null;
  const contentType = response.headers?.get?.('content-type') ?? '';
  if (contentType.includes('application/json')) {
    try {
      body = await response.json();
    } catch {
      body = null;
    }
  }

  return { status: response.status, body };
}

function bearer(token) {
  return { authorization: `Bearer ${token}` };
}

export async function runStagingUserJourney({
  apiUrl,
  fetchImpl = globalThis.fetch,
  identity = createSyntheticIdentity(),
  log = console.log,
}) {
  assert(typeof fetchImpl === 'function', 'fetch implementation is required');
  const baseUrl = normalizeApiUrl(apiUrl);
  const startedAt = Date.now();
  let token = null;
  let accountCreated = false;
  let accountDeleted = false;
  let petId = null;

  const step = async (name, fn) => {
    const start = Date.now();
    const result = await fn();
    log(`staging-journey ${name} ok ${Date.now() - start}ms`);
    return result;
  };

  try {
    const registration = await step('register', () =>
      requestJson(fetchImpl, baseUrl, '/auth/register', {
        method: 'POST',
        body: JSON.stringify({
          email: identity.email,
          handle: identity.handle,
          password: identity.password,
          bio: 'Automated staging release qualification account',
          registrationKey: identity.registrationKey,
        }),
      })
    );
    assert(registration.status === 201, `register expected 201, received ${registration.status}`);
    assert(typeof registration.body?.access_token === 'string', 'register response missing access token');
    assert(registration.body?.user?.email === identity.email, 'register response email mismatch');
    assert(registration.body?.user?.handle === identity.handle, 'register response handle mismatch');
    token = registration.body.access_token;
    accountCreated = true;

    const profile = await step('authenticated-profile', () =>
      requestJson(fetchImpl, baseUrl, '/auth/me', {
        headers: bearer(token),
      })
    );
    assert(profile.status === 200, `auth/me expected 200, received ${profile.status}`);
    assert(profile.body?.email === identity.email, 'auth/me email mismatch');

    const createdPet = await step('owned-pet-create', () =>
      requestJson(fetchImpl, baseUrl, '/pets', {
        method: 'POST',
        headers: bearer(token),
        body: JSON.stringify({
          name: 'Journey Dog',
          species: 'DOG',
          creationKey: identity.petCreationKey,
        }),
      })
    );
    assert(createdPet.status === 201, `pet create expected 201, received ${createdPet.status}`);
    assert(typeof createdPet.body?.id === 'string', 'pet create response missing id');
    petId = createdPet.body.id;

    const updatedPet = await step('owned-pet-update', () =>
      requestJson(fetchImpl, baseUrl, `/pets/${encodeURIComponent(petId)}`, {
        method: 'PUT',
        headers: bearer(token),
        body: JSON.stringify({ breed: 'Synthetic Qualification Dog' }),
      })
    );
    assert(updatedPet.status === 200, `pet update expected 200, received ${updatedPet.status}`);
    assert(updatedPet.body?.id === petId, 'pet update returned a different pet');

    const deletion = await step('account-delete', () =>
      requestJson(fetchImpl, baseUrl, '/users/me', {
        method: 'DELETE',
        headers: bearer(token),
      })
    );
    assert(deletion.status === 200, `account deletion expected 200, received ${deletion.status}`);
    assert(deletion.body?.deleted === true, 'account deletion did not confirm deletion');
    accountDeleted = true;

    const deadSession = await step('deleted-session-rejected', () =>
      requestJson(fetchImpl, baseUrl, '/auth/me', {
        headers: bearer(token),
      })
    );
    assert(deadSession.status === 401, `deleted session expected 401, received ${deadSession.status}`);

    const deadCredentials = await step('deleted-credentials-rejected', () =>
      requestJson(fetchImpl, baseUrl, '/auth/login', {
        method: 'POST',
        body: JSON.stringify({ email: identity.email, password: identity.password }),
      })
    );
    assert(
      deadCredentials.status === 401,
      `deleted credentials expected 401, received ${deadCredentials.status}`
    );

    log(`staging-journey complete ${Date.now() - startedAt}ms`);
    return {
      ok: true,
      accountDeleted: true,
      petAuthorityVerified: true,
      deletedSessionRejected: true,
      deletedCredentialsRejected: true,
    };
  } finally {
    if (accountCreated && !accountDeleted && token) {
      try {
        const cleanup = await requestJson(fetchImpl, baseUrl, '/users/me', {
          method: 'DELETE',
          headers: bearer(token),
        });
        if (cleanup.status === 200 || cleanup.status === 401 || cleanup.status === 404) {
          log('staging-journey cleanup attempted');
        } else {
          console.error(`staging-journey cleanup failed with status ${cleanup.status}`);
        }
      } catch {
        console.error('staging-journey cleanup request failed');
      }
    }
  }
}

function jsonResponse(status, body) {
  return {
    status,
    headers: { get: () => 'application/json' },
    async json() {
      return body;
    },
  };
}

async function selfTest() {
  const identity = {
    email: 'woof.blackbox+selftest@example.com',
    handle: 'bb_selftest',
    password: 'SelfTest!Password123',
    registrationKey: '123e4567-e89b-42d3-a456-426614174000',
    petCreationKey: 'staging-journey:123e4567-e89b-42d3-a456-426614174000',
  };
  const calls = [];
  const successFetch = async (url, options = {}) => {
    const path = new URL(url).pathname;
    calls.push({ path, method: options.method ?? 'GET' });
    if (path.endsWith('/auth/register')) {
      return jsonResponse(201, {
        access_token: 'test-token',
        user: { id: 'user-1', email: identity.email, handle: identity.handle },
      });
    }
    if (path.endsWith('/auth/me') && calls.filter((call) => call.path.endsWith('/auth/me')).length === 1) {
      return jsonResponse(200, { id: 'user-1', email: identity.email, handle: identity.handle });
    }
    if (path.endsWith('/pets') && options.method === 'POST') {
      return jsonResponse(201, { id: 'pet-1', name: 'Journey Dog', species: 'DOG' });
    }
    if (path.endsWith('/pets/pet-1') && options.method === 'PUT') {
      return jsonResponse(200, { id: 'pet-1', breed: 'Synthetic Qualification Dog' });
    }
    if (path.endsWith('/users/me') && options.method === 'DELETE') {
      return jsonResponse(200, { deleted: true });
    }
    if (path.endsWith('/auth/me')) return jsonResponse(401, { statusCode: 401 });
    if (path.endsWith('/auth/login')) return jsonResponse(401, { statusCode: 401 });
    throw new Error(`unexpected self-test request ${options.method ?? 'GET'} ${path}`);
  };

  const result = await runStagingUserJourney({
    apiUrl: 'http://localhost/api/v1',
    fetchImpl: successFetch,
    identity,
    log: () => {},
  });
  assert(result.ok && result.accountDeleted, 'success self-test did not complete');
  assert(calls.some((call) => call.path.endsWith('/users/me') && call.method === 'DELETE'), 'cleanup authority missing');

  let cleanupAttempted = false;
  const failureFetch = async (url, options = {}) => {
    const path = new URL(url).pathname;
    if (path.endsWith('/auth/register')) {
      return jsonResponse(201, {
        access_token: 'failure-token',
        user: { id: 'user-2', email: identity.email, handle: identity.handle },
      });
    }
    if (path.endsWith('/auth/me') && (options.method ?? 'GET') === 'GET') {
      return jsonResponse(500, { statusCode: 500 });
    }
    if (path.endsWith('/users/me') && options.method === 'DELETE') {
      cleanupAttempted = true;
      return jsonResponse(200, { deleted: true });
    }
    throw new Error(`unexpected failure self-test request ${options.method ?? 'GET'} ${path}`);
  };

  let failedAsExpected = false;
  try {
    await runStagingUserJourney({
      apiUrl: 'http://localhost/api/v1',
      fetchImpl: failureFetch,
      identity,
      log: () => {},
    });
  } catch {
    failedAsExpected = true;
  }
  assert(failedAsExpected, 'failure self-test should reject');
  assert(cleanupAttempted, 'failure self-test did not attempt account cleanup');

  console.log('staging user journey verifier self-test passed');
}

async function main() {
  if (process.argv.includes('--self-test')) {
    await selfTest();
    return;
  }
  const apiUrl = process.env.API_URL;
  assert(apiUrl, 'API_URL is required');
  await runStagingUserJourney({ apiUrl });
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error) => {
    console.error(`staging user journey failed: ${error instanceof Error ? error.message : 'unknown error'}`);
    process.exitCode = 1;
  });
}
