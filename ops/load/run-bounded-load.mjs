#!/usr/bin/env node

import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import {
  CONFIG_PATH,
  DEFAULT_REPORT_PATH,
  QualificationError,
  RFC5737_PREFIX,
  SHA_PATTERN,
  expectStatus,
  fail,
  newOperationAccumulator,
  operationEvidence,
  requestJson,
  sleep,
  summarizeDurations,
  writeEvidence,
} from './load-support.mjs';
import {
  connectRealtime,
  finishWorker,
  proveHttpThrottle,
  runMeasuredWorker,
  setupWorker,
} from './load-scenarios.mjs';
import { buildTelemetryEvidence } from './load-telemetry.mjs';

function createReport({ config, profile, profileName, expectedReleaseSha, warnings }) {
  return {
    schemaVersion: 'woof-bounded-load-report-v1',
    harnessVersion: config.harnessVersion,
    generatedAt: new Date().toISOString(),
    environmentClass: config.environmentClass,
    profile: profileName,
    expectedReleaseSha,
    observedReleaseSha: null,
    passed: false,
    failureCodes: [],
    warnings,
    loadProfile: {
      workers: profile.workers,
      durationMs: profile.durationMs,
      requestIntervalMs: profile.requestIntervalMs,
      transitionWaveSize: profile.transitionWaveSize,
      transitionWaves: profile.transitionWaves,
    },
    resourceLimits: config.resourceLimits,
    operations: {},
    realtime: null,
    abuseControl: null,
    telemetry: null,
    invariants: {
      releaseMatched: false,
      syntheticDataOnly: true,
      dailySignalsSingleCanonicalEvent: false,
      dailySignalsDuplicateObserved: false,
      dailySignalsDivergentReplayRejected: false,
      bondXpUnchanged: false,
      caregiverIssueReplaySafe: false,
      caregiverAcceptReplaySafe: false,
      caregiverRevokeReplaySafe: false,
      caregiverDeclineReplaySafe: false,
      caregiverAccessRemoved: false,
      realtimeSessionsReady: false,
      realtimeDisconnectClean: false,
      rateLimit429Observed: false,
    },
  };
}

function attachRuntimeEvidence(report, accumulator, realtimeDurations, profile) {
  report.operations = operationEvidence(accumulator);
  report.realtime = {
    attempts: profile.workers,
    successes: realtimeDurations.length,
    latency: summarizeDurations(realtimeDurations),
  };
}

function requireRepresentativeOperations(report) {
  for (const label of [
    'authMe',
    'adventureMine',
    'companionState',
    'companionReadiness',
    'caregiverToday',
    'healthReady',
  ]) {
    const evidence = report.operations[label];
    if (!evidence || evidence.attempts < 1) fail('REPRESENTATIVE_OPERATION_MISSING');
    if (
      evidence.success2xx !== evidence.attempts ||
      evidence.rateLimited429 !== 0 ||
      evidence.server5xx !== 0 ||
      evidence.client4xx !== 0 ||
      evidence.other !== 0
    ) {
      fail('REPRESENTATIVE_OPERATION_ERROR');
    }
  }
}

async function main() {
  const config = JSON.parse(await readFile(CONFIG_PATH, 'utf8'));
  const profileName = process.env.WOOF_LOAD_PROFILE || 'ci';
  const profile = config.profiles?.[profileName];
  if (!profile) fail('UNKNOWN_LOAD_PROFILE');

  const expectedReleaseSha = (process.env.WOOF_LOAD_EXPECTED_SHA || '').trim().toLowerCase();
  if (!SHA_PATTERN.test(expectedReleaseSha)) fail('EXPECTED_RELEASE_SHA_INVALID');
  const opsToken = process.env.OPS_METRICS_TOKEN;
  if (!opsToken) fail('OPS_METRICS_TOKEN_MISSING');

  const apiOrigin = (process.env.WOOF_LOAD_API_ORIGIN || 'http://127.0.0.1:4000').replace(
    /\/$/,
    ''
  );
  const socketOrigin = (process.env.WOOF_LOAD_SOCKET_ORIGIN || apiOrigin).replace(/\/$/, '');
  const reportPath = resolve(process.env.WOOF_LOAD_REPORT_PATH || DEFAULT_REPORT_PATH);
  const alertPolicy = JSON.parse(
    await readFile(resolve(process.cwd(), config.alertPolicyPath), 'utf8')
  );
  const runTag = randomUUID().replaceAll('-', '').slice(0, 8);
  const accumulator = newOperationAccumulator();
  const realtimeDurations = [];
  const sockets = [];
  const warnings = [];
  const report = createReport({ config, profile, profileName, expectedReleaseSha, warnings });

  try {
    const initialReady = await requestJson({
      apiOrigin,
      path: '/ops/health/ready',
      clientIp: `${RFC5737_PREFIX}249`,
    });
    const initialPayload = expectStatus(initialReady, 200, 'INITIAL_READINESS_FAILED');
    if (initialPayload?.release !== expectedReleaseSha || initialPayload?.status !== 'ready') {
      fail('INITIAL_RELEASE_MISMATCH');
    }
    report.observedReleaseSha = initialPayload.release;
    report.invariants.releaseMatched = true;

    const workers = await Promise.all(
      Array.from({ length: profile.workers }, (_, workerIndex) =>
        setupWorker({ apiOrigin, workerIndex, runTag, profile, accumulator })
      )
    );
    report.invariants.dailySignalsSingleCanonicalEvent = true;
    report.invariants.dailySignalsDuplicateObserved = true;
    report.invariants.dailySignalsDivergentReplayRejected = true;
    report.invariants.caregiverIssueReplaySafe = true;
    report.invariants.caregiverAcceptReplaySafe = true;

    const realtimeConnections = await Promise.all(
      workers.map((worker) => connectRealtime({ socketOrigin, worker }))
    );
    for (const connection of realtimeConnections) {
      sockets.push(connection.socket);
      realtimeDurations.push(connection.durationMs);
    }
    report.invariants.realtimeSessionsReady = realtimeConnections.length === workers.length;

    await Promise.all(
      workers.map((worker) => runMeasuredWorker({ apiOrigin, worker, profile, accumulator }))
    );

    for (const socket of sockets) socket.disconnect();
    await sleep(50);
    report.invariants.realtimeDisconnectClean = sockets.every((socket) => !socket.connected);

    await Promise.all(
      workers.map((worker) => finishWorker({ apiOrigin, worker, runTag, profile, accumulator }))
    );
    report.invariants.caregiverRevokeReplaySafe = true;
    report.invariants.caregiverDeclineReplaySafe = true;
    report.invariants.caregiverAccessRemoved = true;
    report.invariants.bondXpUnchanged = true;

    const abuseControl = await proveHttpThrottle({ apiOrigin });
    report.abuseControl = abuseControl;
    report.invariants.rateLimit429Observed = abuseControl.rateLimited429 > 0;

    await sleep(profile.concurrencyResetMs);
    const metricsResponse = await requestJson({
      apiOrigin,
      path: '/ops/metrics.json',
      clientIp: `${RFC5737_PREFIX}248`,
      opsToken,
    });
    const metrics = expectStatus(metricsResponse, 200, 'METRICS_SNAPSHOT_FAILED');
    report.observedReleaseSha = metrics?.release ?? report.observedReleaseSha;
    report.telemetry = buildTelemetryEvidence({
      snapshot: metrics,
      alertPolicy,
      profile,
      expectedReleaseSha,
      warnings,
    });

    attachRuntimeEvidence(report, accumulator, realtimeDurations, profile);
    requireRepresentativeOperations(report);
    if (!Object.values(report.invariants).every((value) => value === true)) {
      fail('INVARIANT_NOT_PROVEN');
    }

    report.passed = true;
    await writeEvidence(reportPath, report);
    console.log('bounded operational load qualification passed');
  } catch (error) {
    for (const socket of sockets) socket.disconnect();
    attachRuntimeEvidence(report, accumulator, realtimeDurations, profile);
    const code = error instanceof QualificationError ? error.code : 'HARNESS_ERROR';
    report.failureCodes = [code];
    await writeEvidence(reportPath, report);
    console.error(`bounded operational load qualification failed: ${code}`);
    process.exitCode = 1;
  }
}

main().catch(() => {
  console.error('bounded operational load qualification failed: HARNESS_BOOTSTRAP_ERROR');
  process.exitCode = 1;
});
