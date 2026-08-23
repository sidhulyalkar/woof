import {
  ServiceUnavailableException,
  UnauthorizedException,
  type ExecutionContext,
} from '@nestjs/common';
import { OpsTokenGuard } from './ops-token.guard';

function context(token?: string) {
  return {
    switchToHttp: () => ({
      getRequest: () => ({ headers: token ? { 'x-woof-ops-token': token } : {} }),
    }),
  } as unknown as ExecutionContext;
}

describe('OpsTokenGuard', () => {
  const original = process.env.OPS_METRICS_TOKEN;

  afterEach(() => {
    if (original === undefined) delete process.env.OPS_METRICS_TOKEN;
    else process.env.OPS_METRICS_TOKEN = original;
  });

  it('fails closed when operational metrics authentication is not configured', () => {
    delete process.env.OPS_METRICS_TOKEN;
    const guard = new OpsTokenGuard();

    expect(() => guard.canActivate(context())).toThrow(ServiceUnavailableException);
  });

  it('rejects missing and incorrect credentials', () => {
    process.env.OPS_METRICS_TOKEN = 'correct-secret';
    const guard = new OpsTokenGuard();

    expect(() => guard.canActivate(context())).toThrow(UnauthorizedException);
    expect(() => guard.canActivate(context('wrong'))).toThrow(UnauthorizedException);
  });

  it('accepts only the configured credential', () => {
    process.env.OPS_METRICS_TOKEN = 'correct-secret';
    const guard = new OpsTokenGuard();

    expect(guard.canActivate(context('correct-secret'))).toBe(true);
  });
});
