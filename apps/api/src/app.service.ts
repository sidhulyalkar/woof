import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from './prisma/prisma.service';

@Injectable()
export class AppService {
  constructor(private readonly prisma: PrismaService) {}

  getInfo() {
    const isProduction = process.env.NODE_ENV === 'production';
    const docsEnabled = !isProduction || process.env.API_DOCS_ENABLED === 'true';

    return {
      name: 'Woof API',
      version: '1.0.0',
      description: 'Pet Social Fitness Platform - Galaxy Dark Edition',
      docs: docsEnabled ? '/docs' : null,
      endpoints: {
        auth: '/api/v1/auth',
        users: '/api/v1/users',
        pets: '/api/v1/pets',
        activities: '/api/v1/activities',
        social: '/api/v1/social',
        meetups: '/api/v1/meetups',
        compatibility: '/api/v1/compatibility',
      },
    };
  }

  getLiveness() {
    return {
      status: 'alive',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
    };
  }

  async getReadiness() {
    const startedAt = Date.now();

    try {
      await this.prisma.$queryRaw<Array<{ ready: number }>>(Prisma.sql`SELECT 1::int AS ready`);
      return {
        status: 'ready',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
        checks: {
          database: {
            status: 'up',
            latencyMs: Date.now() - startedAt,
          },
        },
      };
    } catch {
      throw new ServiceUnavailableException({
        status: 'not_ready',
        timestamp: new Date().toISOString(),
        checks: {
          database: {
            status: 'down',
          },
        },
      });
    }
  }

  getHealth() {
    return this.getReadiness();
  }
}
