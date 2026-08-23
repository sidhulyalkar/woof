import { OperationalMetricsService } from './operational-metrics.service';

describe('OperationalMetricsService', () => {
  it('stores only low-cardinality operational labels and aggregate timings', () => {
    const service = new OperationalMetricsService();

    service.recordRequest({
      method: 'get',
      operation: 'PetsController.findOwned',
      statusCode: 200,
      durationMs: 12.5,
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
            count: 1,
            durationMsTotal: 12.5,
            durationMsMax: 12.5,
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

  it('exports aggregate Prometheus metrics without request URLs or entity identifiers', () => {
    const service = new OperationalMetricsService();
    service.recordRequest({
      method: 'POST',
      operation: 'ConnectorsController.disconnect',
      statusCode: 401,
      durationMs: 3,
    });

    const metrics = service.prometheus();

    expect(metrics).toContain('operation="ConnectorsController.disconnect"');
    expect(metrics).toContain('status_class="4xx"');
    expect(metrics).not.toContain('/api/');
    expect(metrics).not.toContain('userId');
    expect(metrics).not.toContain('petId');
    expect(metrics).not.toContain('externalObjectId');
  });
});
