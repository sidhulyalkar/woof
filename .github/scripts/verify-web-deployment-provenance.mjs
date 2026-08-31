#!/usr/bin/env node

/**
 * Verify public Woof Web deployment provenance without exposing private config.
 *
 * Inputs are intentionally explicit:
 *   WEB_HTML          rendered HTML from a public deployment
 *   EXPECTED_RELEASE  exact 40-hex Git SHA
 *   EXPECTED_API_URL  public API base URL compiled into the Web artifact
 */

const GIT_SHA_PATTERN = /^[0-9a-f]{40}$/;

function decodeHtml(value) {
  return value
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&amp;', '&');
}

function attributes(tag) {
  const result = new Map();
  const pattern = /([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
  for (const match of tag.matchAll(pattern)) {
    result.set(match[1].toLowerCase(), decodeHtml(match[2] ?? match[3] ?? ''));
  }
  return result;
}

export function metaValue(html, name) {
  for (const match of html.matchAll(/<meta\b[^>]*>/gi)) {
    const attrs = attributes(match[0]);
    if (attrs.get('name') === name) return attrs.get('content') ?? null;
  }
  return null;
}

export function verifyWebDeploymentProvenance({ html, expectedRelease, expectedApiUrl }) {
  const normalizedRelease = expectedRelease.trim().toLowerCase();
  if (!GIT_SHA_PATTERN.test(normalizedRelease)) {
    throw new Error(
      `Expected release must be one exact 40-hex Git SHA, received '${expectedRelease}'.`
    );
  }

  let apiUrl;
  try {
    apiUrl = new URL(expectedApiUrl);
  } catch {
    throw new Error(`Expected API URL is invalid: '${expectedApiUrl}'.`);
  }
  if (apiUrl.protocol !== 'https:') {
    throw new Error(`Expected API URL must use HTTPS, received '${expectedApiUrl}'.`);
  }

  const release = metaValue(html, 'woof-release');
  const apiOrigin = metaValue(html, 'woof-api-origin');
  if (release !== normalizedRelease) {
    throw new Error(`Web release mismatch: expected ${normalizedRelease}, received ${release}.`);
  }
  if (apiOrigin !== expectedApiUrl) {
    throw new Error(`Web API origin mismatch: expected ${expectedApiUrl}, received ${apiOrigin}.`);
  }

  return { release, apiOrigin };
}

function selfTest() {
  const release = '0123456789abcdef0123456789abcdef01234567';
  const api = 'https://api.example.com/api/v1';

  const variants = [
    `<html><head><meta name="woof-release" content="${release}"><meta name="woof-api-origin" content="${api}"></head></html>`,
    `<meta content='${release}' data-extra='x' name='woof-release'><meta content='${api}' name='woof-api-origin'>`,
    `<meta content="${release}" name="woof-release"/><meta name="woof-api-origin" content="https://api.example.com/api/v1"/>`,
  ];
  for (const html of variants) {
    verifyWebDeploymentProvenance({ html, expectedRelease: release, expectedApiUrl: api });
  }

  for (const [label, input] of [
    [
      'wrong release',
      {
        html: variants[0].replace(release, 'f'.repeat(40)),
        expectedRelease: release,
        expectedApiUrl: api,
      },
    ],
    [
      'wrong API',
      {
        html: variants[0].replace(api, 'https://wrong.example.com/api/v1'),
        expectedRelease: release,
        expectedApiUrl: api,
      },
    ],
    [
      'missing markers',
      { html: '<html></html>', expectedRelease: release, expectedApiUrl: api },
    ],
  ]) {
    let rejected = false;
    try {
      verifyWebDeploymentProvenance(input);
    } catch {
      rejected = true;
    }
    if (!rejected) throw new Error(`Self-test failed to reject ${label}.`);
  }

  console.log('Web deployment provenance verifier self-test passed.');
}

if (process.argv.includes('--self-test')) {
  selfTest();
} else if (import.meta.url === `file://${process.argv[1]}`) {
  const html = process.env.WEB_HTML ?? '';
  const expectedRelease = process.env.EXPECTED_RELEASE ?? '';
  const expectedApiUrl = process.env.EXPECTED_API_URL ?? '';
  const result = verifyWebDeploymentProvenance({ html, expectedRelease, expectedApiUrl });
  console.log(`Verified Woof Web deployment ${result.release} -> ${result.apiOrigin}`);
}
