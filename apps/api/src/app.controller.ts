import { Controller, Get } from '@nestjs/common';
import { ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import { AppService } from './app.service';

@ApiTags('health')
@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get()
  @ApiOperation({ summary: 'Root endpoint' })
  @ApiResponse({ status: 200, description: 'API info' })
  getInfo() {
    return this.appService.getInfo();
  }

  @Get('health')
  @ApiOperation({ summary: 'Compatibility readiness check' })
  @ApiResponse({ status: 200, description: 'Service is ready to receive traffic' })
  @ApiResponse({ status: 503, description: 'A required dependency is unavailable' })
  getHealth() {
    return this.appService.getHealth();
  }

  @Get('health/live')
  @ApiOperation({ summary: 'Process liveness check' })
  @ApiResponse({ status: 200, description: 'API process is alive' })
  getLiveness() {
    return this.appService.getLiveness();
  }

  @Get('health/ready')
  @ApiOperation({ summary: 'Dependency readiness check' })
  @ApiResponse({ status: 200, description: 'Service dependencies are ready' })
  @ApiResponse({ status: 503, description: 'A required dependency is unavailable' })
  getReadiness() {
    return this.appService.getReadiness();
  }
}
