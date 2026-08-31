import { OperationalMetricsService, REQUEST_DURATION_BUCKETS_MS } from './operational-metrics.service';

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

  it('normalizes non-finite durations instead of poisoning aggregate telemetry', () => {
    const service = new OperationalMetricsService();
    service.recordRequest({
      method: 'GET',
      operation: 'ObservabilityController.readiness',
      statusCode: 503,
      durationMs: Number.NaN,
    });

    const [request] = service.snapshot().requests;
    expect(request.durationMsTotal).toBe(0);
    expect(request.durationMsMax).toBe(0);
    expect(request.durationMsBuckets.every((count) => count === 1)).toBe(true);
    expect(service.prometheus()).not.toContain('NaN');
  });
});
