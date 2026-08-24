import { RealtimeAdmissionService } from './realtime-admission.service';

describe('RealtimeAdmissionService', () => {
  it('limits rapid message attempts before persistence work', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 5; index += 1) {
      expect(limiter.consume('user-1', 'message', 1_000)).toEqual({ allowed: true });
    }

    expect(limiter.consume('user-1', 'message', 1_000)).toEqual({
      allowed: false,
      retryAfterMs: 1_000,
    });
    expect(limiter.consume('user-1', 'message', 2_000)).toEqual({ allowed: true });
  });

  it('keeps authenticated users in independent realtime buckets', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 5; index += 1) {
      expect(limiter.consume('user-1', 'message', 1_000)).toEqual({ allowed: true });
    }

    expect(limiter.consume('user-1', 'message', 1_000).allowed).toBe(false);
    expect(limiter.consume('user-2', 'message', 1_000)).toEqual({ allowed: true });
  });

  it('bounds sustained message volume across the long window', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 60; index += 1) {
      expect(limiter.consume('user-1', 'message', index * 1_001)).toEqual({ allowed: true });
    }

    const denied = limiter.consume('user-1', 'message', 59_060);
    expect(denied.allowed).toBe(false);
    if (!denied.allowed) {
      expect(denied.retryAfterMs).toBeGreaterThan(0);
    }
  });

  it('cuts off typing floods before repeated authorization work', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 8; index += 1) {
      expect(limiter.consume('user-1', 'typing', 5_000)).toEqual({ allowed: true });
    }

    expect(limiter.consume('user-1', 'typing', 5_000)).toEqual({
      allowed: false,
      retryAfterMs: 5_000,
    });
  });

  it('shares one user bucket across reconnect-style repeated calls', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 10; index += 1) {
      expect(limiter.consume('user-1', 'membership', 10_000)).toEqual({ allowed: true });
    }

    expect(limiter.consume('user-1', 'membership', 10_000).allowed).toBe(false);
  });

  it('evicts old identities when the bounded bucket cap is reached', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 5; index += 1) {
      limiter.consume('old-user', 'message', 0);
    }

    for (let index = 0; index < 10_000; index += 1) {
      limiter.consume(`user-${index}`, 'message', 0);
    }

    expect(limiter.consume('old-user', 'message', 0)).toEqual({ allowed: true });
  });

  it('expires inactive window state', () => {
    const limiter = new RealtimeAdmissionService();

    for (let index = 0; index < 5; index += 1) {
      limiter.consume('user-1', 'message', 0);
    }

    expect(limiter.consume('user-1', 'message', 60_001)).toEqual({ allowed: true });
  });
});
