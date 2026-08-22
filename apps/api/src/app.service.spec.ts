import { ServiceUnavailableException } from '@nestjs/common';
import { PrismaService } from './prisma/prisma.service';
import { AppService } from './app.service';

function makeService() {
  const prisma = {
    $queryRaw: jest.fn(),
  };
  return {
    prisma,
    service: new AppService(prisma as unknown as PrismaService),
  };
}

describe('AppService health checks', () => {
  it('reports process liveness without touching dependencies', () => {
    const { prisma, service } = makeService();

    expect(service.getLiveness()).toEqual(
      expect.objectContaining({
        status: 'alive',
        timestamp: expect.any(String),
        uptime: expect.any(Number),
      }),
    );
    expect(prisma.$queryRaw).not.toHaveBeenCalled();
  });

  it('reports readiness only after a real database probe succeeds', async () => {
    const { prisma, service } = makeService();
    prisma.$queryRaw.mockResolvedValue([{ ready: 1 }]);

    await expect(service.getReadiness()).resolves.toEqual(
      expect.objectContaining({
        status: 'ready',
        checks: {
          database: expect.objectContaining({
            status: 'up',
            latencyMs: expect.any(Number),
          }),
        },
      }),
    );
    expect(prisma.$queryRaw).toHaveBeenCalledTimes(1);
  });

  it('fails readiness closed when the database cannot be reached', async () => {
    const { prisma, service } = makeService();
    prisma.$queryRaw.mockRejectedValue(new Error('database unavailable'));

    await expect(service.getReadiness()).rejects.toBeInstanceOf(ServiceUnavailableException);
  });
});
