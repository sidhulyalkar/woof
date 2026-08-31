import { Injectable } from '@nestjs/common';
import type { ConnectorProvider } from '../connectors/connectors.types';
import { resolveReleaseIdentity } from './release-identity';

export const REQUEST_DURATION_BUCKETS_MS = [
  25,
  50,
  100,
  250,
  500,
  750,
  1000,
  1500,
  2500,
  5000,
] as const;

type RequestMetric = {
  method: string;
  operation: string;
  statusClass: string;
  count: number;
  durationSampleCount: number;
  durationInvalidCount: number;
  durationMsTotal: number;
  durationMsMax: number;
  durationMsBuckets: number[];
};

type ConnectorMetric = {
  provider: ConnectorProvider;
  kind: 'DAILY_ACTIVITY' | 'DEVICE_STATUS';
  outcome: 'IMPORTED' | 'DUPLICATE' | 'REJECTED';
  count: number;
  durationSampleCount: number;
  durationInvalidCount: number;
  durationMsTotal: number;
  durationMsMax: number;
};

function escapePrometheus(value: string) {
  return value.replaceAll('\\', '\\\\').replaceAll('"', '\\"').replaceAll('\n', '\\n');
}

function statusClass(statusCode: number) {
  if (!Number.isFinite(statusCode)) return '5xx';
  const normalized = Math.floor(statusCode);
  if (normalized < 100 || normalized > 599) return '5xx';
  return `${Math.floor(normalized / 100)}xx`;
}

function validDurationMs(durationMs: number) {
  return Number.isFinite(durationMs) && durationMs >= 0 ? durationMs : null;
}

@Injectable()
export class OperationalMetricsService {
  private readonly startedAt = new Date();
  private readonly release = resolveReleaseIdentity();
  private readonly requests = new Map<string, RequestMetric>();
  private readonly connectorImports = new Map<string, ConnectorMetric>();
  private deviceContractRejections = 0;

  recordRequest(input: {
    method: string;
    operation: string;
    statusCode: number;
    durationMs: number;
  }) {
    const method = input.method.toUpperCase().slice(0, 12);
    const operation = input.operation.slice(0, 160);
    const bucket = statusClass(input.statusCode);
    const durationMs = validDurationMs(input.durationMs);
    const key = `${method}\u0000${operation}\u0000${bucket}`;
    const current = this.requests.get(key) ?? {
      method,
      operation,
      statusClass: bucket,
      count: 0,
      durationSampleCount: 0,
      durationInvalidCount: 0,
      durationMsTotal: 0,
      durationMsMax: 0,
      durationMsBuckets: REQUEST_DURATION_BUCKETS_MS.map(() => 0),
    };

    current.count += 1;
    if (durationMs === null) {
      current.durationInvalidCount += 1;
    } else {
      current.durationSampleCount += 1;
      current.durationMsTotal += durationMs;
      current.durationMsMax = Math.max(current.durationMsMax, durationMs);
      REQUEST_DURATION_BUCKETS_MS.forEach((upperBound, index) => {
        if (durationMs <= upperBound) current.durationMsBuckets[index] += 1;
      });
    }
    this.requests.set(key, current);
  }

  recordConnectorImport(input: {
    provider: ConnectorProvider;
    kind: 'DAILY_ACTIVITY' | 'DEVICE_STATUS';
    outcome: 'IMPORTED' | 'DUPLICATE' | 'REJECTED';
    durationMs: number;
  }) {
    const durationMs = validDurationMs(input.durationMs);
    const key = `${input.provider}\u0000${input.kind}\u0000${input.outcome}`;
    const current = this.connectorImports.get(key) ?? {
      provider: input.provider,
      kind: input.kind,
      outcome: input.outcome,
      count: 0,
      durationSampleCount: 0,
      durationInvalidCount: 0,
      durationMsTotal: 0,
      durationMsMax: 0,
    };

    current.count += 1;
    if (durationMs === null) {
      current.durationInvalidCount += 1;
    } else {
      current.durationSampleCount += 1;
      current.durationMsTotal += durationMs;
      current.durationMsMax = Math.max(current.durationMsMax, durationMs);
    }
    this.connectorImports.set(key, current);
  }

  recordDeviceContractRejection() {
    this.deviceContractRejections += 1;
  }

  snapshot() {
    return {
      scope: 'process' as const,
      service: 'woof-api' as const,
      release: this.release,
      startedAt: this.startedAt.toISOString(),
      uptimeSeconds: process.uptime(),
      privacy: {
        userIdentifiersCollected: false as const,
        petIdentifiersCollected: false as const,
        providerExternalIdentifiersCollected: false as const,
        rawPayloadsCollected: false as const,
        requestUrlsCollected: false as const,
      },
      requestDurationBucketsMs: [...REQUEST_DURATION_BUCKETS_MS],
      deviceContractRejections: this.deviceContractRejections,
      requests: [...this.requests.values()].sort((left, right) =>
        `${left.method}:${left.operation}:${left.statusClass}`.localeCompare(
          `${right.method}:${right.operation}:${right.statusClass}`
        )
      ),
      connectorImports: [...this.connectorImports.values()].sort((left, right) =>
        `${left.provider}:${left.kind}:${left.outcome}`.localeCompare(
          `${right.provider}:${right.kind}:${right.outcome}`
        )
      ),
    };
  }

  prometheus() {
    const baseLabels = `service="woof-api",release="${escapePrometheus(this.release)}"`;
    const lines = [
      '# HELP woof_release_info Exact qualified release identity exposed by this process.',
      '# TYPE woof_release_info gauge',
      `woof_release_info{${baseLabels}} 1`,
      '# HELP woof_process_uptime_seconds Process uptime in seconds.',
      '# TYPE woof_process_uptime_seconds gauge',
      `woof_process_uptime_seconds{${baseLabels}} ${process.uptime()}`,
      '# HELP woof_http_requests_total Requests handled by operation and status class.',
      '# TYPE woof_http_requests_total counter',
      '# HELP woof_http_request_duration_ms Request handler duration histogram by operation and status class.',
      '# TYPE woof_http_request_duration_ms histogram',
      '# HELP woof_http_request_duration_invalid_total Request timing samples excluded because duration was non-finite or negative.',
      '# TYPE woof_http_request_duration_invalid_total counter',
    ];

    for (const metric of this.requests.values()) {
      const labels = `${baseLabels},method="${escapePrometheus(metric.method)}",operation="${escapePrometheus(
        metric.operation
      )}",status_class="${escapePrometheus(metric.statusClass)}"`;
      lines.push(`woof_http_requests_total{${labels}} ${metric.count}`);
      metric.durationMsBuckets.forEach((count, index) => {
        lines.push(
          `woof_http_request_duration_ms_bucket{${labels},le="${REQUEST_DURATION_BUCKETS_MS[index]}"} ${count}`
        );
      });
      lines.push(
        `woof_http_request_duration_ms_bucket{${labels},le="+Inf"} ${metric.durationSampleCount}`
      );
      lines.push(`woof_http_request_duration_ms_sum{${labels}} ${metric.durationMsTotal}`);
      lines.push(`woof_http_request_duration_ms_count{${labels}} ${metric.durationSampleCount}`);
      lines.push(
        `woof_http_request_duration_invalid_total{${labels}} ${metric.durationInvalidCount}`
      );
    }

    lines.push(
      '# HELP woof_device_contract_rejections_total Device envelopes rejected before trusted provider labels are available.',
      '# TYPE woof_device_contract_rejections_total counter',
      `woof_device_contract_rejections_total{${baseLabels}} ${this.deviceContractRejections}`,
      '# HELP woof_connector_imports_total Verified device imports by provider, kind and outcome.',
      '# TYPE woof_connector_imports_total counter',
      '# HELP woof_connector_import_duration_invalid_total Connector timing samples excluded because duration was non-finite or negative.',
      '# TYPE woof_connector_import_duration_invalid_total counter'
    );
    for (const metric of this.connectorImports.values()) {
      const labels = `${baseLabels},provider="${metric.provider}",kind="${metric.kind}",outcome="${metric.outcome}"`;
      lines.push(`woof_connector_imports_total{${labels}} ${metric.count}`);
      lines.push(`woof_connector_import_duration_ms_sum{${labels}} ${metric.durationMsTotal}`);
      lines.push(
        `woof_connector_import_duration_ms_count{${labels}} ${metric.durationSampleCount}`
      );
      lines.push(`woof_connector_import_duration_ms_max{${labels}} ${metric.durationMsMax}`);
      lines.push(
        `woof_connector_import_duration_invalid_total{${labels}} ${metric.durationInvalidCount}`
      );
    }

    return `${lines.join('\n')}\n`;
  }
}
