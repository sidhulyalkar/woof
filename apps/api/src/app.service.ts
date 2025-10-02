import { Injectable } from '@nestjs/common';

@Injectable()
export class AppService {
  getInfo() {
    return {
      name: 'Woof API',
      version: '1.0.0',
      description: 'Pet Social Fitness Platform - Galaxy Dark Edition',
      docs: '/docs',
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

  getHealth() {
    return {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      database: 'connected', // Will be updated when we add real health checks
    };
  }
}
