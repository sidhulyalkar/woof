import {
  OperationalMetricsService,
  REQUEST_DURATION_BUCKETS_MS,
} from './operational-metrics.service';

const RELEASE_SHA = 'a'.repeat(40);

describe('OperationalMetricsService', () => {
  const previousRelease = process.env.WOOF_RELEASE_SHA;

  beforeEach(() => {
    process.env.WOOF_RELEASE_SHA = RELEASE_SHA;
  });

  afterAll(() => {
    if (previousRelease === undefined) delete process.env.WOOF_RELEASE_SHA;
    else process.env.WOOF_RELEASE_SHA = previousRelease;
  });

  it('stores only low-cardinality operational labels and cumulative timing buckets', () => {
    const service = new OperationalMetricsService();

    service.recordRequest({
      method: 'get',
      operation: 'PetsController.findOwned',
      statusCode: 200,
      durationMs: 12.5,
    });
    service.recordRequest({
      method: 'get',
      operation: 'PetsController.findOwned',
      statusCode: 200,
      durationMs: 800,
    });
    service.recordConnectorImport({
      provider: 'TRACTIVE',
      kind: 'DAILY_ACTIVITY',
      outcome: 'IMPORTED',
      durationMs: 42,
    });

    expect(service.snapshot()).toEqual(
      expect.objectContaining({
        scope: 'process',
        service: 'woof-api',
        release: RELEASE_SHA,
        requestDurationBucketsMs: [...REQUEST_DURATION_BUCKETS_MS],
        privacy: {
          userIdentifiersCollected: false,
          petIdentifiersCollected: false,
          providerExternalIdentifiersCollected: false,
          rawPayloadsCollected: false,
          requestUrlsCollected: false,
        },
        requests: [
          expect.objectContaining({
            method: 'GET',
            operation: 'PetsController.findOwned',
            statusClass: '2xx',
            count: 2,
            durationSampleCount: 2,
            durationInvalidCount: 0,
            durationMsTotal: 812.5,
            durationMsMax: 800,
            durationMsBuckets: [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
          }),
        ],
        connectorImports: [
          expect.objectContaining({
            provider: 'TRACTIVE',
            kind: 'DAILY_ACTIVITY',
            outcome: 'IMPORTED',
            count: 1,
            durationSampleCount: 1,
            durationInvalidCount: 0,
          }),
        ],
      })
    );
  });

  it('exports release-labeled Prometheus histograms without request URLs or entity identifiers', () => {
    const service = new OperationalMetricsService();
    service.recordRequest({
      method: 'POST',
      operation: 'ConnectorsController.disconnect',
      statusCode: 401,
      durationMs: 800,
    });

    const metrics = service.prometheus();

    expect(metrics).toContain(`woof_release_info{service="woof-api",release="${RELEASE_SHA}"} 1`);
    expect(metrics).toContain('operation="ConnectorsController.disconnect"');
    expect(metrics).toContain('status_class="4xx"');
    expect(metrics).toContain('woof_http_request_duration_ms_bucket');
    expect(metrics).toContain('le="750"} 0');
    expect(metrics).toContain('le="1000"} 1');
    expect(metrics).toContain('le="+Inf"} 1');
    expect(metrics).not.toContain('/api/');
    expect(metrics).not.toContain('userId');
    expect(metrics).not.toContain('petId');
    expect(metrics).not.toContain('externalObjectId');
  });

  it('excludes invalid durations from latency histograms instead of recording fake zeroes', () => {
    const service = new OperationalMetricsService();
    service.recordRequest({
      method: 'GET',
      operation: 'ObservabilityController.readiness',
      statusCode: 503,
      durationMs: Number.NaN,
    });
    service.recordConnectorImport({
      provider: 'TRACTIVE',
      kind: 'DEVICE_STATUS',
      outcome: 'REJECTED',
      durationMs: Number.NEGATIVE_INFINITY,
    });

    const snapshot = service.snapshot();
    const [request] = snapshot.requests;
    const [connector] = snapshot.connectorImports;
    expect(request.count).toBe(1);
    expect(request.durationSampleCount).toBe(0);
    expect(request.durationInvalidCount).toBe(1);
    expect(request.durationMsTotal).toBe(0);
    expect(request.durationMsBuckets.every((count) => count === 0)).toBe(true);
    expect(connector.count).toBe(1);
    expect(connector.durationSampleCount).toBe(0);
    expect(connector.durationInvalidCount).toBe(1);

    const metrics = service.prometheus();
    expect(metrics).toContain('woof_http_request_duration_invalid_total');
    expect(metrics).toContain('le="+Inf"} 0');
    expect(metrics).toContain('woof_connector_import_duration_invalid_total');
    expect(metrics).not.toContain('NaN');
    expect(metrics).not.toContain('Infinity');
  });

  it('fails malformed HTTP status codes into the 5xx operational class', () => {
    const service = new OperationalMetricsService();
    service.recordRequest({
      method: 'GET',
      operation: 'ExampleController.invalidStatus',
      statusCode: 999,
      durationMs: 1,
    });

    expect(service.snapshot().requests[0].statusClass).toBe('5xx');
  });
});
