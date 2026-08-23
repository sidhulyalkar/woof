import { timingSafeEqual } from 'node:crypto';
import {
  CanActivate,
  ExecutionContext,
  Injectable,
  ServiceUnavailableException,
  UnauthorizedException,
} from '@nestjs/common';

function constantTimeEqual(left: string, right: string) {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  if (leftBuffer.length !== rightBuffer.length) return false;
  return timingSafeEqual(leftBuffer, rightBuffer);
}

@Injectable()
export class OpsTokenGuard implements CanActivate {
  canActivate(context: ExecutionContext) {
    const configured = process.env.OPS_METRICS_TOKEN?.trim();
    if (!configured) {
      throw new ServiceUnavailableException('Operational metrics are not configured');
    }

    const request = context.switchToHttp().getRequest<{ headers?: Record<string, unknown> }>();
    const supplied = request.headers?.['x-woof-ops-token'];
    if (typeof supplied !== 'string' || !constantTimeEqual(supplied, configured)) {
      throw new UnauthorizedException('Invalid operational metrics credential');
    }
    return true;
  }
}
