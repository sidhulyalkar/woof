import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { resolveProcessReleaseIdentity } from './release-identity';

@Injectable()
export class ObservabilityService {
  constructor(private readonly prisma: PrismaService) {}

  liveness() {
    return {
      status: 'live' as const,
      timestamp: new Date().toISOString(),
      uptimeSeconds: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      release: resolveProcessReleaseIdentity(),
    };
  }

  async readiness() {
    const started = performance.now();
    const release = resolveProcessReleaseIdentity();
    try {
      await this.prisma.$queryRaw`SELECT 1`;
      return {
        status: 'ready' as const,
        timestamp: new Date().toISOString(),
        release,
        database: {
          status: 'ready' as const,
          latencyMs: Math.max(0, performance.now() - started),
        },
      };
    } catch {
      return {
        status: 'not_ready' as const,
        timestamp: new Date().toISOString(),
        release,
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
