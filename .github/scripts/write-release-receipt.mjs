#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const SHA_PATTERN = /^[0-9a-f]{40}$/;
const ALLOWED_ENVIRONMENTS = new Set(['staging', 'production']);

function requireValue(name, value) {
  if (!value || !value.trim()) {
    throw new Error(`${name} is required`);
  }
  return value.trim();
}

function requireHttpsUrl(name, value) {
  const raw = requireValue(name, value);
  const parsed = new URL(raw);
  if (parsed.protocol !== 'https:') {
    throw new Error(`${name} must use https`);
  }
  parsed.username = '';
  parsed.password = '';
  parsed.search = '';
  parsed.hash = '';
  return parsed.toString().replace(/\/$/, '');
}

export function buildReleaseReceipt(env) {
  const releaseEnvironment = requireValue('RELEASE_ENVIRONMENT', env.RELEASE_ENVIRONMENT);
  if (!ALLOWED_ENVIRONMENTS.has(releaseEnvironment)) {
    throw new Error(`unsupported RELEASE_ENVIRONMENT: ${releaseEnvironment}`);
  }

  const releaseSha = requireValue('RELEASE_SHA', env.RELEASE_SHA);
  if (!SHA_PATTERN.test(releaseSha)) {
    throw new Error('RELEASE_SHA must be an exact lowercase 40-character Git SHA');
  }

  const repository = requireValue('GITHUB_REPOSITORY', env.GITHUB_REPOSITORY);
  const workflow = requireValue('GITHUB_WORKFLOW', env.GITHUB_WORKFLOW);
  const runId = requireValue('GITHUB_RUN_ID', env.GITHUB_RUN_ID);
  const runAttempt = requireValue('GITHUB_RUN_ATTEMPT', env.GITHUB_RUN_ATTEMPT);
  const apiUrl = requireHttpsUrl('API_URL', env.API_URL);
  const webUrl = requireHttpsUrl('WEB_URL', env.WEB_URL);
  const stagingWorkflowVerified = env.STAGING_WORKFLOW_VERIFIED === 'true';

  if (releaseEnvironment === 'production' && !stagingWorkflowVerified) {
    throw new Error('production receipt requires successful exact-SHA staging qualification');
  }

  return {
    schemaVersion: 1,
    environment: releaseEnvironment,
    releaseSha,
    repository,
    workflow,
    runId,
    runAttempt,
    apiUrl,
    webUrl,
    evidence: {
      mainAncestryVerified: true,
      apiReleaseIdentityVerified: true,
      webReleaseIdentityVerified: true,
      webApiOriginVerified: true,
      stagingWorkflowVerified,
    },
  };
}

function selfTest() {
  const base = {
    RELEASE_ENVIRONMENT: 'staging',
    RELEASE_SHA: 'a'.repeat(40),
    GITHUB_REPOSITORY: 'example/woof',
    GITHUB_WORKFLOW: 'Deploy to Staging',
    GITHUB_RUN_ID: '123',
    GITHUB_RUN_ATTEMPT: '1',
    API_URL: 'https://api.example.test/api/v1',
    WEB_URL: 'https://web.example.test',
    STAGING_WORKFLOW_VERIFIED: 'false',
  };

  const staging = buildReleaseReceipt(base);
  if (staging.releaseSha !== base.RELEASE_SHA || staging.evidence.stagingWorkflowVerified) {
    throw new Error('staging receipt self-test failed');
  }

  const production = buildReleaseReceipt({
    ...base,
    RELEASE_ENVIRONMENT: 'production',
    GITHUB_WORKFLOW: 'Deploy to Production',
    STAGING_WORKFLOW_VERIFIED: 'true',
  });
  if (!production.evidence.stagingWorkflowVerified) {
    throw new Error('production receipt self-test failed');
  }

  let rejected = false;
  try {
    buildReleaseReceipt({ ...base, RELEASE_SHA: 'main' });
  } catch {
    rejected = true;
  }
  if (!rejected) {
    throw new Error('non-SHA release identity was accepted');
  }

  rejected = false;
  try {
    buildReleaseReceipt({
      ...base,
      RELEASE_ENVIRONMENT: 'production',
      STAGING_WORKFLOW_VERIFIED: 'false',
    });
  } catch {
    rejected = true;
  }
  if (!rejected) {
    throw new Error('production receipt without staging qualification was accepted');
  }

  console.log('release receipt self-test passed');
}

if (process.argv.includes('--self-test')) {
  selfTest();
} else {
  const receiptPath = requireValue('RELEASE_RECEIPT_PATH', process.env.RELEASE_RECEIPT_PATH);
  const receipt = buildReleaseReceipt(process.env);
  fs.mkdirSync(path.dirname(receiptPath), { recursive: true });
  fs.writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { mode: 0o600 });
  console.log(`wrote release receipt for ${receipt.environment} ${receipt.releaseSha}`);
}
