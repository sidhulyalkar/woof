import {
  CallHandler,
  ExecutionContext,
  Injectable,
  NestInterceptor,
  type HttpException,
} from '@nestjs/common';
import type { Observable } from 'rxjs';
import { tap } from 'rxjs/operators';
import { OperationalMetricsService } from './operational-metrics.service';

function exceptionStatus(error: unknown) {
  if (error && typeof error === 'object' && 'getStatus' in error) {
    const getStatus = (error as Pick<HttpException, 'getStatus'>).getStatus;
    if (typeof getStatus === 'function') return getStatus.call(error);
  }
  return 500;
}

@Injectable()
export class RequestMetricsInterceptor implements NestInterceptor {
  constructor(private readonly metrics: OperationalMetricsService) {}

  intercept(context: ExecutionContext, next: CallHandler): Observable<unknown> {
    if (context.getType() !== 'http') return next.handle();

    const request = context.switchToHttp().getRequest<{ method?: string }>();
    const response = context.switchToHttp().getResponse<{ statusCode?: number }>();
    const operation = `${context.getClass().name}.${context.getHandler().name}`;
    const method = request.method ?? 'UNKNOWN';
    const started = performance.now();
    let recorded = false;

    const record = (statusCode: number) => {
      if (recorded) return;
      recorded = true;
      this.metrics.recordRequest({
        method,
        operation,
        statusCode,
        durationMs: Math.max(0, performance.now() - started),
      });
    };

    return next.handle().pipe(
      tap({
        next: () => record(response.statusCode ?? 200),
        error: (error) => record(exceptionStatus(error)),
      })
    );
  }
}
