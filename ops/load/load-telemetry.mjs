import { fail } from './load-support.mjs';

function aggregateMetricRows(snapshot, operation) {
  const rows = Array.isArray(snapshot?.requests)
    ? snapshot.requests.filter((row) => row?.operation === operation)
    : [];
  const successRows = rows.filter((row) => row?.statusClass === '2xx');
  const bucketCount = Array.isArray(snapshot?.requestDurationBucketsMs)
    ? snapshot.requestDurationBucketsMs.length
    : 0;
  const durationMsBuckets = Array.from({ length: bucketCount }, (_, index) =>
    successRows.reduce((sum, row) => sum + Number(row?.durationMsBuckets?.[index] ?? 0), 0)
  );
  return {
    success2xx: successRows.reduce((sum, row) => sum + Number(row?.count ?? 0), 0),
    server5xx: rows
      .filter((row) => row?.statusClass === '5xx')
      .reduce((sum, row) => sum + Number(row?.count ?? 0), 0),
    durationSampleCount: successRows.reduce(
      (sum, row) => sum + Number(row?.durationSampleCount ?? 0),
      0
    ),
    durationInvalidCount: rows.reduce(
      (sum, row) => sum + Number(row?.durationInvalidCount ?? 0),
      0
    ),
    durationMsBuckets,
  };
}

function histogramUpperBound(snapshot, metric, quantile) {
  if (metric.durationSampleCount < 1) return null;
  const target = Math.ceil(metric.durationSampleCount * quantile);
  for (let index = 0; index < metric.durationMsBuckets.length; index += 1) {
    if (metric.durationMsBuckets[index] >= target) return snapshot.requestDurationBucketsMs[index];
  }
  return 'INF';
}

function todayClassification(upperBound, warningMs, criticalMs) {
  if (upperBound === 'INF' || upperBound === null || upperBound > criticalMs) return 'CRITICAL';
  if (upperBound > warningMs) return 'WARNING';
  return 'OK';
}

function requirePrivacySnapshot(snapshot) {
  const privacy = snapshot?.privacy;
  if (
    !privacy ||
    privacy.userIdentifiersCollected !== false ||
    privacy.petIdentifiersCollected !== false ||
    privacy.providerExternalIdentifiersCollected !== false ||
    privacy.rawPayloadsCollected !== false ||
    privacy.requestUrlsCollected !== false
  ) {
    fail('METRICS_PRIVACY_CONTRACT');
  }
}

export function buildTelemetryEvidence({
  snapshot,
  alertPolicy,
  profile,
  expectedReleaseSha,
  warnings,
}) {
  if (snapshot?.release !== expectedReleaseSha) fail('METRICS_RELEASE_MISMATCH');
  requirePrivacySnapshot(snapshot);

  const todayPolicy = alertPolicy.todayReadP95Ms;
  const todayReads = {};
  for (const operation of todayPolicy.operations) {
    const metric = aggregateMetricRows(snapshot, operation);
    if (metric.durationSampleCount < todayPolicy.minimumRequests) fail('TODAY_READ_SAMPLE_FLOOR');
    if (metric.server5xx !== 0 || metric.durationInvalidCount !== 0) {
      fail('TODAY_READ_ERROR_EVIDENCE');
    }
    const p95BucketUpperBoundMs = histogramUpperBound(snapshot, metric, todayPolicy.quantile);
    const classification = todayClassification(
      p95BucketUpperBoundMs,
      todayPolicy.warningMs,
      todayPolicy.criticalMs
    );
    if (classification === 'CRITICAL') fail('TODAY_READ_CRITICAL_LATENCY');
    if (classification === 'WARNING') warnings.push(`TODAY_READ_WARNING:${operation}`);
    todayReads[operation] = {
      samples: metric.durationSampleCount,
      p95BucketUpperBoundMs,
      classification,
      server5xx: metric.server5xx,
      durationInvalid: metric.durationInvalidCount,
    };
  }

  const authMetric = aggregateMetricRows(snapshot, 'AuthController.getProfile');
  if (
    authMetric.durationSampleCount < profile.minimumAuthSamples ||
    authMetric.server5xx !== 0 ||
    authMetric.durationInvalidCount !== 0
  ) {
    fail('AUTH_SESSION_LOAD_EVIDENCE');
  }

  const readinessMetric = aggregateMetricRows(snapshot, 'ObservabilityController.readiness');
  if (
    readinessMetric.durationSampleCount < profile.minimumReadinessSamples ||
    readinessMetric.server5xx !== 0 ||
    readinessMetric.durationInvalidCount !== 0
  ) {
    fail('READINESS_LOAD_EVIDENCE');
  }

  const caregiverTransitions = {};
  for (const operation of alertPolicy.caregiverTransition5xx.operations) {
    const metric = aggregateMetricRows(snapshot, operation);
    if (metric.success2xx < alertPolicy.caregiverTransition5xx.minimumRequests) {
      fail('CAREGIVER_TRANSITION_SAMPLE_FLOOR');
    }
    if (metric.server5xx !== 0 || metric.durationInvalidCount !== 0) {
      fail('CAREGIVER_TRANSITION_ERROR_EVIDENCE');
    }
    caregiverTransitions[operation] = {
      samples: metric.durationSampleCount,
      success2xx: metric.success2xx,
      server5xx: metric.server5xx,
      durationInvalid: metric.durationInvalidCount,
    };
  }

  const allRows = Array.isArray(snapshot.requests) ? snapshot.requests : [];
  const totalServer5xx = allRows
    .filter((row) => row?.statusClass === '5xx')
    .reduce((sum, row) => sum + Number(row?.count ?? 0), 0);
  const totalDurationInvalid = allRows.reduce(
    (sum, row) => sum + Number(row?.durationInvalidCount ?? 0),
    0
  );
  if (totalServer5xx !== 0) fail('UNEXPECTED_SERVER_5XX');
  if (totalDurationInvalid !== 0) fail('INVALID_DURATION_SAMPLE');

  return {
    metricsScope: snapshot.scope,
    privacyContractPassed: true,
    todayReads,
    authSession: {
      samples: authMetric.durationSampleCount,
      server5xx: authMetric.server5xx,
      durationInvalid: authMetric.durationInvalidCount,
    },
    readiness: {
      samples: readinessMetric.durationSampleCount,
      server5xx: readinessMetric.server5xx,
      durationInvalid: readinessMetric.durationInvalidCount,
    },
    caregiverTransitions,
    totalServer5xx,
    totalDurationInvalid,
  };
}
