import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

export const CONFIG_PATH = resolve(process.cwd(), 'ops/load/woof-bounded-load.v1.json');
export const DEFAULT_REPORT_PATH = resolve(process.cwd(), 'artifacts/operational-load/report.json');
export const SHA_PATTERN = /^[0-9a-f]{40}$/;
export const RFC5737_PREFIX = '203.0.113.';
const API_PREFIX = '/api/v1';

export const sleep = (ms) => new Promise((resolvePromise) => setTimeout(resolvePromise, ms));

export class QualificationError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

export function fail(code) {
  throw new QualificationError(code);
}

function machineCode(value) {
  return String(value)
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[^A-Za-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toUpperCase();
}

function statusBucket(status) {
  if (status === 429) return 'rateLimited429';
  if (status >= 200 && status < 300) return 'success2xx';
  if (status >= 400 && status < 500) return 'client4xx';
  if (status >= 500 && status < 600) return 'server5xx';
  return 'other';
}

function percentile(values, quantile) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(ordered.length * quantile) - 1);
  return Math.round(ordered[index] * 100) / 100;
}

export function summarizeDurations(values) {
  return {
    samples: values.length,
    p50Ms: percentile(values, 0.5),
    p95Ms: percentile(values, 0.95),
    p99Ms: percentile(values, 0.99),
    maxMs: values.length ? Math.round(Math.max(...values) * 100) / 100 : null,
  };
}

export function newOperationAccumulator() {
  return new Map();
}

function recordOperation(accumulator, label, result) {
  const current = accumulator.get(label) ?? {
    attempts: 0,
    success2xx: 0,
    client4xx: 0,
    rateLimited429: 0,
    server5xx: 0,
    other: 0,
    durationsMs: [],
  };
  current.attempts += 1;
  current[statusBucket(result.status)] += 1;
  current.durationsMs.push(result.durationMs);
  accumulator.set(label, current);
}

export function operationEvidence(accumulator) {
  return Object.fromEntries(
    [...accumulator.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([label, value]) => [
        label,
        {
          attempts: value.attempts,
          success2xx: value.success2xx,
          client4xx: value.client4xx,
          rateLimited429: value.rateLimited429,
          server5xx: value.server5xx,
          other: value.other,
          latency: summarizeDurations(value.durationsMs),
        },
      ])
  );
}

function safeJson(payload) {
  return payload && typeof payload === 'object' ? payload : null;
}

export async function requestJson({
  apiOrigin,
  path,
  method = 'GET',
  clientIp,
  token,
  opsToken,
  body,
  operationLabel,
  accumulator,
}) {
  const started = performance.now();
  let response;
  try {
    response = await fetch(`${apiOrigin}${API_PREFIX}${path}`, {
      method,
      headers: {
        'fly-client-ip': clientIp,
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        ...(opsToken ? { 'x-woof-ops-token': opsToken } : {}),
        ...(body === undefined ? {} : { 'content-type': 'application/json' }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  } catch {
    fail(`NETWORK_${machineCode(operationLabel ?? 'REQUEST')}`);
  }

  const durationMs = Math.max(0, performance.now() - started);
  let payload = null;
  try {
    payload = safeJson(await response.json());
  } catch {
    payload = null;
  }

  const result = { status: response.status, durationMs, payload };
  if (operationLabel && accumulator) recordOperation(accumulator, operationLabel, result);
  return result;
}

export function expectStatus(result, expected, code) {
  if (result.status !== expected) fail(code);
  return result.payload;
}

export function createSyntheticClient({ apiOrigin, ip, token, setupPaceMs, accumulator }) {
  return {
    apiOrigin,
    ip,
    token,
    async paced(path, options = {}) {
      await sleep(setupPaceMs);
      return requestJson({
        apiOrigin,
        path,
        clientIp: ip,
        token: options.token === undefined ? token : options.token,
        method: options.method,
        body: options.body,
        operationLabel: options.operationLabel,
        accumulator,
      });
    },
  };
}

export async function replayWaves({
  client,
  path,
  method = 'POST',
  body,
  operationLabel,
  profile,
  expectedStatus,
  accumulator,
}) {
  const results = [];
  for (let wave = 0; wave < profile.transitionWaves; wave += 1) {
    await sleep(profile.concurrencyResetMs);
    const waveResults = await Promise.all(
      Array.from({ length: profile.transitionWaveSize }, () =>
        requestJson({
          apiOrigin: client.apiOrigin,
          path,
          clientIp: client.ip,
          token: client.token,
          method,
          body,
          operationLabel,
          accumulator,
        })
      )
    );
    for (const result of waveResults) {
      if (result.status !== expectedStatus) {
        fail(`STATUS_${machineCode(operationLabel)}_${result.status}`);
      }
      results.push(result);
    }
  }
  return results;
}

export function requireSameIdentity(results, field, code) {
  const values = results.map((result) => result.payload?.[field]);
  if (values.some((value) => typeof value !== 'string' || !value)) fail(code);
  if (new Set(values).size !== 1) fail(code);
  return values[0];
}

export function requireReplayObserved(results, code) {
  if (!results.some((result) => result.payload?.replayed === true)) fail(code);
}

export async function writeEvidence(reportPath, report) {
  await mkdir(dirname(reportPath), { recursive: true });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
}
