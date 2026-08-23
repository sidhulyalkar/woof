import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class ObservabilityService {
  constructor(private readonly prisma: PrismaService) {}

  liveness() {
    return {
      status: 'live' as const,
      timestamp: new Date().toISOString(),
      uptimeSeconds: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
    };
  }

  async readiness() {
    const started = performance.now();
    try {
      await this.prisma.$queryRaw`SELECT 1`;
      return {
        status: 'ready' as const,
        timestamp: new Date().toISOString(),
        database: {
          status: 'ready' as const,
          latencyMs: Math.max(0, performance.now() - started),
        },
      };
    } catch {
      return {
        status: 'not_ready' as const,
        timestamp: new Date().toISOString(),
        database: {
          status: 'unavailable' as const,
          latencyMs: Math.max(0, performance.now() - started),
        },
      };
    }
  }

  async assertReady() {
    const readiness = await this.readiness();
    if (readiness.status !== 'ready') {
      throw new ServiceUnavailableException(readiness);
    }
    return readiness;
  }

  async health() {
    const readiness = await this.readiness();
    return {
      ...this.liveness(),
      status: readiness.status === 'ready' ? ('healthy' as const) : ('degraded' as const),
      readiness,
    };
  }
}
