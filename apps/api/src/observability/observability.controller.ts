import { Controller, Get, Header, UseGuards } from '@nestjs/common';
import { ApiExcludeController } from '@nestjs/swagger';
import { ObservabilityService } from './observability.service';
import { OperationalMetricsService } from './operational-metrics.service';
import { OpsTokenGuard } from './ops-token.guard';

@ApiExcludeController()
@Controller('ops')
export class ObservabilityController {
  constructor(
    private readonly metrics: OperationalMetricsService,
    private readonly observability: ObservabilityService
  ) {}

  @Get('health/live')
  liveness() {
    return this.observability.liveness();
  }

  @Get('health/ready')
  readiness() {
    return this.observability.assertReady();
  }

  @Get('metrics')
  @UseGuards(OpsTokenGuard)
  @Header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
  prometheus() {
    return this.metrics.prometheus();
  }

  @Get('metrics.json')
  @UseGuards(OpsTokenGuard)
  snapshot() {
    return this.metrics.snapshot();
  }
}
