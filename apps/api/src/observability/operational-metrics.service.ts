import { Injectable } from '@nestjs/common';
import type { ConnectorProvider } from '../connectors/connectors.types';

type RequestMetric = {
  method: string;
  operation: string;
  statusClass: string;
  count: number;
  durationMsTotal: number;
  durationMsMax: number;
};

type ConnectorMetric = {
  provider: ConnectorProvider;
  kind: 'DAILY_ACTIVITY' | 'DEVICE_STATUS';
  outcome: 'IMPORTED' | 'DUPLICATE' | 'REJECTED';
  count: number;
  durationMsTotal: number;
  durationMsMax: number;
};

function escapePrometheus(value: string) {
  return value.replaceAll('\\', '\\\\').replaceAll('"', '\\"').replaceAll('\n', '\\n');
}

function statusClass(statusCode: number) {
  const normalized = Number.isFinite(statusCode) ? Math.max(0, Math.floor(statusCode)) : 500;
  return `${Math.floor(normalized / 100)}xx`;
}

@Injectable()
export class OperationalMetricsService {
  private readonly startedAt = new Date();
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
    const key = `${method}\u0000${operation}\u0000${bucket}`;
    const current = this.requests.get(key) ?? {
      method,
      operation,
      statusClass: bucket,
      count: 0,
      durationMsTotal: 0,
      durationMsMax: 0,
    };
    current.count += 1;
    current.durationMsTotal += input.durationMs;
    current.durationMsMax = Math.max(current.durationMsMax, input.durationMs);
    this.requests.set(key, current);
  }

  recordConnectorImport(input: {
    provider: ConnectorProvider;
    kind: 'DAILY_ACTIVITY' | 'DEVICE_STATUS';
    outcome: 'IMPORTED' | 'DUPLICATE' | 'REJECTED';
    durationMs: number;
  }) {
    const key = `${input.provider}\u0000${input.kind}\u0000${input.outcome}`;
    const current = this.connectorImports.get(key) ?? {
      provider: input.provider,
      kind: input.kind,
      outcome: input.outcome,
      count: 0,
      durationMsTotal: 0,
      durationMsMax: 0,
    };
    current.count += 1;
    current.durationMsTotal += input.durationMs;
    current.durationMsMax = Math.max(current.durationMsMax, input.durationMs);
    this.connectorImports.set(key, current);
  }

  recordDeviceContractRejection() {
    this.deviceContractRejections += 1;
  }

  snapshot() {
    return {
      scope: 'process' as const,
      startedAt: this.startedAt.toISOString(),
      uptimeSeconds: process.uptime(),
      privacy: {
        userIdentifiersCollected: false as const,
        petIdentifiersCollected: false as const,
        providerExternalIdentifiersCollected: false as const,
        rawPayloadsCollected: false as const,
        requestUrlsCollected: false as const,
      },
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
    const lines = [
      '# HELP woof_process_uptime_seconds Process uptime in seconds.',
      '# TYPE woof_process_uptime_seconds gauge',
      `woof_process_uptime_seconds ${process.uptime()}`,
      '# HELP woof_http_requests_total Requests handled by operation and status class.',
      '# TYPE woof_http_requests_total counter',
    ];

    for (const metric of this.requests.values()) {
      const labels = `method="${escapePrometheus(metric.method)}",operation="${escapePrometheus(
        metric.operation
      )}",status_class="${escapePrometheus(metric.statusClass)}"`;
      lines.push(`woof_http_requests_total{${labels}} ${metric.count}`);
      lines.push(`woof_http_request_duration_ms_sum{${labels}} ${metric.durationMsTotal}`);
      lines.push(`woof_http_request_duration_ms_count{${labels}} ${metric.count}`);
      lines.push(`woof_http_request_duration_ms_max{${labels}} ${metric.durationMsMax}`);
    }

    lines.push(
      '# HELP woof_device_contract_rejections_total Device envelopes rejected before trusted provider labels are available.',
      '# TYPE woof_device_contract_rejections_total counter',
      `woof_device_contract_rejections_total ${this.deviceContractRejections}`,
      '# HELP woof_connector_imports_total Verified device imports by provider, kind and outcome.',
      '# TYPE woof_connector_imports_total counter'
    );
    for (const metric of this.connectorImports.values()) {
      const labels = `provider="${metric.provider}",kind="${metric.kind}",outcome="${metric.outcome}"`;
      lines.push(`woof_connector_imports_total{${labels}} ${metric.count}`);
      lines.push(`woof_connector_import_duration_ms_sum{${labels}} ${metric.durationMsTotal}`);
      lines.push(`woof_connector_import_duration_ms_count{${labels}} ${metric.count}`);
      lines.push(`woof_connector_import_duration_ms_max{${labels}} ${metric.durationMsMax}`);
    }

    return `${lines.join('\n')}\n`;
  }
}
