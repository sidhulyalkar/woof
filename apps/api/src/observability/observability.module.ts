import { Global, Module } from '@nestjs/common';
import { APP_INTERCEPTOR } from '@nestjs/core';
import { ObservabilityController } from './observability.controller';
import { ObservabilityService } from './observability.service';
import { OperationalMetricsService } from './operational-metrics.service';
import { OpsTokenGuard } from './ops-token.guard';
import { RequestMetricsInterceptor } from './request-metrics.interceptor';

@Global()
@Module({
  controllers: [ObservabilityController],
  providers: [
    ObservabilityService,
    OperationalMetricsService,
    OpsTokenGuard,
    {
      provide: APP_INTERCEPTOR,
      useClass: RequestMetricsInterceptor,
    },
  ],
  exports: [ObservabilityService, OperationalMetricsService],
})
export class ObservabilityModule {}
