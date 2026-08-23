import { ServiceUnavailableException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { ObservabilityService } from './observability.service';

function harness() {
  const prisma = {
    $queryRaw: jest.fn().mockResolvedValue([{ ok: 1 }]),
  };
  return {
    prisma,
    service: new ObservabilityService(prisma as unknown as PrismaService),
  };
}

describe('ObservabilityService', () => {
  it('keeps liveness independent from database readiness', () => {
    const { service, prisma } = harness();

    expect(service.liveness()).toEqual(
      expect.objectContaining({ status: 'live', uptimeSeconds: expect.any(Number) })
    );
    expect(prisma.$queryRaw).not.toHaveBeenCalled();
  });

  it('reports database readiness only after a real query succeeds', async () => {
    const { service, prisma } = harness();

    await expect(service.readiness()).resolves.toEqual(
      expect.objectContaining({
        status: 'ready',
        database: expect.objectContaining({
          status: 'ready',
          latencyMs: expect.any(Number),
        }),
      })
    );
    expect(prisma.$queryRaw).toHaveBeenCalledTimes(1);
  });

  it('returns degraded health and fails the readiness assertion when the database is unavailable', async () => {
    const { service, prisma } = harness();
    prisma.$queryRaw.mockRejectedValue(new Error('database unavailable'));

    await expect(service.health()).resolves.toEqual(
      expect.objectContaining({
        status: 'degraded',
        readiness: expect.objectContaining({
          status: 'not_ready',
          database: expect.objectContaining({ status: 'unavailable' }),
        }),
      })
    );
    await expect(service.assertReady()).rejects.toBeInstanceOf(ServiceUnavailableException);
  });
});
