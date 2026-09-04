import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
import {
  QualificationError,
  RFC5737_PREFIX,
  createSyntheticClient,
  expectStatus,
  fail,
  replayWaves,
  requestJson,
  requireReplayObserved,
  requireSameIdentity,
  sleep,
} from './load-support.mjs';

const requireFromWeb = createRequire(new URL('../../apps/web/package.json', import.meta.url));
const { io } = requireFromWeb('socket.io-client');

async function registerSynthetic({
  apiOrigin,
  ip,
  role,
  workerIndex,
  runTag,
  profile,
  accumulator,
}) {
  const client = createSyntheticClient({
    apiOrigin,
    ip,
    setupPaceMs: profile.setupPaceMs,
    accumulator,
  });
  const response = await client.paced('/auth/register', {
    method: 'POST',
    body: {
      handle: `ld-${runTag}-${role}-${workerIndex}`.slice(0, 30),
      email: `woof-load-${runTag}-${role}-${workerIndex}@example.test`,
      password: 'LoadOnlyPass123!',
      registrationKey: randomUUID(),
    },
  });
  const payload = expectStatus(response, 201, 'REGISTER_SYNTHETIC_FAILED');
  if (!payload?.access_token || typeof payload?.user?.id !== 'string') {
    fail('REGISTER_SYNTHETIC_CONTRACT');
  }
  return {
    client: createSyntheticClient({
      apiOrigin,
      ip,
      token: payload.access_token,
      setupPaceMs: profile.setupPaceMs,
      accumulator,
    }),
    userId: payload.user.id,
  };
}

export async function setupWorker({ apiOrigin, workerIndex, runTag, profile, accumulator }) {
  const owner = await registerSynthetic({
    apiOrigin,
    ip: `${RFC5737_PREFIX}${1 + workerIndex * 2}`,
    role: 'owner',
    workerIndex,
    runTag,
    profile,
    accumulator,
  });
  const caregiver = await registerSynthetic({
    apiOrigin,
    ip: `${RFC5737_PREFIX}${2 + workerIndex * 2}`,
    role: 'care',
    workerIndex,
    runTag,
    profile,
    accumulator,
  });

  const petResponse = await owner.client.paced('/pets', {
    method: 'POST',
    body: {
      name: `Load dog ${workerIndex}`,
      species: 'DOG',
      creationKey: `bounded-load-v1:${runTag}:${workerIndex}`,
    },
  });
  const pet = expectStatus(petResponse, 201, 'PET_CREATE_FAILED');
  if (typeof pet?.id !== 'string') fail('PET_CREATE_CONTRACT');

  const householdsResponse = await owner.client.paced('/households/me');
  const households = expectStatus(householdsResponse, 200, 'HOUSEHOLD_READ_FAILED');
  if (!Array.isArray(households)) fail('HOUSEHOLD_READ_CONTRACT');
  const household = households.find(
    (candidate) =>
      candidate?.viewerRole === 'OWNER' &&
      Array.isArray(candidate?.pets) &&
      candidate.pets.some((link) => link?.pet?.id === pet.id)
  );
  if (typeof household?.id !== 'string') fail('HOUSEHOLD_OWNER_NOT_FOUND');

  const timezoneResponse = await owner.client.paced(`/households/${household.id}`, {
    method: 'PATCH',
    body: { timezone: 'UTC' },
  });
  expectStatus(timezoneResponse, 200, 'HOUSEHOLD_TIMEZONE_FAILED');

  const activeGrantBody = {
    petId: pet.id,
    recipientUserId: caregiver.userId,
    capabilities: ['VIEW_TODAY'],
    expiresAt: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
    requestKey: `load-active-${runTag}-${workerIndex}`,
  };
  const activeIssue = await replayWaves({
    client: owner.client,
    path: '/caregiver/grants',
    body: activeGrantBody,
    operationLabel: 'caregiverIssue',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  const activeGrantId = requireSameIdentity(activeIssue, 'id', 'CAREGIVER_ISSUE_IDENTITY');
  requireReplayObserved(activeIssue, 'CAREGIVER_ISSUE_REPLAY');

  const accepts = await replayWaves({
    client: caregiver.client,
    path: `/caregiver/grants/${activeGrantId}/accept`,
    body: {},
    operationLabel: 'caregiverAccept',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  requireSameIdentity(accepts, 'id', 'CAREGIVER_ACCEPT_IDENTITY');
  requireReplayObserved(accepts, 'CAREGIVER_ACCEPT_REPLAY');
  if (accepts.some((result) => result.payload?.effectiveStatus !== 'ACTIVE')) {
    fail('CAREGIVER_ACCEPT_STATUS');
  }

  await sleep(profile.concurrencyResetMs);
  const baselineAdventure = await requestJson({
    apiOrigin,
    path: `/adventure/me?petId=${encodeURIComponent(pet.id)}`,
    clientIp: owner.client.ip,
    token: owner.client.token,
    operationLabel: 'adventureBaseline',
    accumulator,
  });
  const baselinePayload = expectStatus(baselineAdventure, 200, 'ADVENTURE_BASELINE_FAILED');
  if (typeof baselinePayload?.bondXp !== 'number' || !Number.isFinite(baselinePayload.bondXp)) {
    fail('ADVENTURE_BASELINE_CONTRACT');
  }

  const dailySignalsBody = {
    householdId: household.id,
    petId: pet.id,
    signals: {
      appetite: 'USUAL',
      energy: 'USUAL',
      bathroomRoutine: 'USUAL',
      mobilityComfort: 'USUAL',
      engagementSocialComfort: 'USUAL',
      sleepRest: 'USUAL',
    },
  };
  const dailySignals = await replayWaves({
    client: owner.client,
    path: '/intelligence/daily-signals',
    body: dailySignalsBody,
    operationLabel: 'dailySignalsRetry',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  requireSameIdentity(dailySignals, 'careEventId', 'DAILY_SIGNALS_CANONICAL_IDENTITY');
  if (!dailySignals.some((result) => result.payload?.duplicate === true)) {
    fail('DAILY_SIGNALS_DUPLICATE_NOT_OBSERVED');
  }
  if (dailySignals.filter((result) => result.payload?.duplicate === false).length !== 1) {
    fail('DAILY_SIGNALS_PRIMARY_COUNT');
  }

  await sleep(profile.concurrencyResetMs);
  const divergent = await requestJson({
    apiOrigin,
    path: '/intelligence/daily-signals',
    clientIp: owner.client.ip,
    token: owner.client.token,
    method: 'POST',
    body: {
      ...dailySignalsBody,
      signals: { ...dailySignalsBody.signals, energy: 'MORE' },
    },
    operationLabel: 'dailySignalsConflict',
    accumulator,
  });
  if (divergent.status !== 409) fail('DAILY_SIGNALS_DIVERGENT_NOT_REJECTED');

  await sleep(profile.concurrencyResetMs);
  const caregiverToday = await requestJson({
    apiOrigin,
    path: `/caregiver/pets/${encodeURIComponent(pet.id)}/today`,
    clientIp: caregiver.client.ip,
    token: caregiver.client.token,
  });
  expectStatus(caregiverToday, 200, 'CAREGIVER_TODAY_PRELOAD_FAILED');

  return {
    workerIndex,
    owner: owner.client,
    caregiver: caregiver.client,
    caregiverUserId: caregiver.userId,
    petId: pet.id,
    activeGrantId,
    baselineBondXp: baselinePayload.bondXp,
  };
}

export function connectRealtime({ socketOrigin, worker }) {
  return new Promise((resolvePromise, reject) => {
    const started = performance.now();
    const socket = io(socketOrigin, {
      auth: { token: worker.owner.token },
      transports: ['websocket'],
      reconnection: false,
      timeout: 5000,
      extraHeaders: { 'fly-client-ip': worker.owner.ip },
    });
    const timer = setTimeout(() => {
      socket.disconnect();
      reject(new QualificationError('REALTIME_SESSION_READY_TIMEOUT'));
    }, 6000);

    socket.once('session:ready', (payload) => {
      clearTimeout(timer);
      if (!payload || payload.socketId !== socket.id) {
        socket.disconnect();
        reject(new QualificationError('REALTIME_SESSION_READY_MISMATCH'));
        return;
      }
      resolvePromise({ socket, durationMs: Math.max(0, performance.now() - started) });
    });
    socket.once('connect_error', () => {
      clearTimeout(timer);
      socket.disconnect();
      reject(new QualificationError('REALTIME_CONNECT_ERROR'));
    });
  });
}

export async function runMeasuredWorker({ apiOrigin, worker, profile, accumulator }) {
  const operations = [
    { label: 'authMe', client: worker.owner, path: () => '/auth/me' },
    {
      label: 'adventureMine',
      client: worker.owner,
      path: () => `/adventure/me?petId=${encodeURIComponent(worker.petId)}`,
    },
    { label: 'companionState', client: worker.owner, path: () => '/companion/state' },
    {
      label: 'companionReadiness',
      client: worker.owner,
      path: () => '/companion/readiness',
    },
    {
      label: 'caregiverToday',
      client: worker.caregiver,
      path: () => `/caregiver/pets/${encodeURIComponent(worker.petId)}/today`,
    },
    {
      label: 'healthReady',
      client: worker.owner,
      path: () => '/ops/health/ready',
      token: null,
    },
  ];

  const started = performance.now();
  const deadline = started + profile.durationMs;
  let nextStart = started;
  let iteration = 0;
  while (performance.now() < deadline) {
    const operation = operations[(iteration + worker.workerIndex) % operations.length];
    const result = await requestJson({
      apiOrigin,
      path: operation.path(),
      clientIp: operation.client.ip,
      token: operation.token === null ? undefined : operation.client.token,
      operationLabel: operation.label,
      accumulator,
    });
    if (result.status < 200 || result.status >= 300) fail(`MEASURED_${operation.label}_STATUS`);
    iteration += 1;
    nextStart += profile.requestIntervalMs;
    const pause = nextStart - performance.now();
    if (pause > 0) await sleep(pause);
  }
}

export async function finishWorker({ apiOrigin, worker, runTag, profile, accumulator }) {
  const revokes = await replayWaves({
    client: worker.owner,
    path: `/caregiver/grants/${worker.activeGrantId}/revoke`,
    body: {},
    operationLabel: 'caregiverRevoke',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  requireSameIdentity(revokes, 'id', 'CAREGIVER_REVOKE_IDENTITY');
  requireReplayObserved(revokes, 'CAREGIVER_REVOKE_REPLAY');
  if (revokes.some((result) => result.payload?.effectiveStatus !== 'REVOKED')) {
    fail('CAREGIVER_REVOKE_STATUS');
  }

  await sleep(profile.concurrencyResetMs);
  const revokedToday = await requestJson({
    apiOrigin,
    path: `/caregiver/pets/${encodeURIComponent(worker.petId)}/today`,
    clientIp: worker.caregiver.ip,
    token: worker.caregiver.token,
  });
  if (![403, 404].includes(revokedToday.status)) fail('CAREGIVER_REVOKED_ACCESS');

  const declinedGrantBody = {
    petId: worker.petId,
    recipientUserId: worker.caregiverUserId,
    capabilities: ['VIEW_TODAY'],
    expiresAt: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
    requestKey: `load-decline-${runTag}-${worker.workerIndex}`,
  };
  const declinedIssue = await replayWaves({
    client: worker.owner,
    path: '/caregiver/grants',
    body: declinedGrantBody,
    operationLabel: 'caregiverIssue',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  const declinedGrantId = requireSameIdentity(
    declinedIssue,
    'id',
    'CAREGIVER_DECLINE_ISSUE_IDENTITY'
  );
  requireReplayObserved(declinedIssue, 'CAREGIVER_DECLINE_ISSUE_REPLAY');

  const declines = await replayWaves({
    client: worker.caregiver,
    path: `/caregiver/grants/${declinedGrantId}/decline`,
    body: {},
    operationLabel: 'caregiverDecline',
    profile,
    expectedStatus: 201,
    accumulator,
  });
  requireSameIdentity(declines, 'id', 'CAREGIVER_DECLINE_IDENTITY');
  requireReplayObserved(declines, 'CAREGIVER_DECLINE_REPLAY');
  if (declines.some((result) => result.payload?.effectiveStatus !== 'DECLINED')) {
    fail('CAREGIVER_DECLINE_STATUS');
  }

  await sleep(profile.concurrencyResetMs);
  const finalAdventure = await requestJson({
    apiOrigin,
    path: `/adventure/me?petId=${encodeURIComponent(worker.petId)}`,
    clientIp: worker.owner.ip,
    token: worker.owner.token,
    operationLabel: 'adventureFinal',
    accumulator,
  });
  const finalPayload = expectStatus(finalAdventure, 200, 'ADVENTURE_FINAL_FAILED');
  if (finalPayload?.bondXp !== worker.baselineBondXp) fail('BOND_XP_CHANGED_UNDER_LOAD');
}

export async function proveHttpThrottle({ apiOrigin }) {
  const abuseIp = `${RFC5737_PREFIX}250`;
  await sleep(1100);
  const results = await Promise.all(
    Array.from({ length: 6 }, () =>
      requestJson({
        apiOrigin,
        path: '/ops/health/live',
        clientIp: abuseIp,
      })
    )
  );
  const success2xx = results.filter((result) => result.status >= 200 && result.status < 300).length;
  const rateLimited429 = results.filter((result) => result.status === 429).length;
  if (success2xx < 1 || rateLimited429 < 1) fail('HTTP_THROTTLE_NOT_OBSERVED');
  return { attempts: results.length, success2xx, rateLimited429 };
}
