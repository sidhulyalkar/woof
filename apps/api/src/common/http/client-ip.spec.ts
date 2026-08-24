import { clientIpTrackerForEnv, flyClientIp, isTrustedFlyRuntime } from './client-ip';

describe('clientIpTracker', () => {
  const flyEnv = {
    FLY_APP_NAME: 'woof-api-ci',
    FLY_MACHINE_ID: 'machine-ci',
  } as NodeJS.ProcessEnv;

  it('trusts a single valid Fly-Client-IP only inside a proven Fly runtime', async () => {
    await expect(
      clientIpTrackerForEnv(
        {
          headers: { 'fly-client-ip': '203.0.113.10' },
          ip: '172.16.0.10',
        },
        flyEnv
      )
    ).resolves.toBe('fly:203.0.113.10');
  });

  it('keeps distinct Fly clients in distinct throttle identities', async () => {
    const [clientA, clientB] = await Promise.all([
      clientIpTrackerForEnv(
        { headers: { 'fly-client-ip': '203.0.113.10' }, ip: '172.16.0.10' },
        flyEnv
      ),
      clientIpTrackerForEnv(
        { headers: { 'fly-client-ip': '203.0.113.11' }, ip: '172.16.0.10' },
        flyEnv
      ),
    ]);

    expect(clientA).toBe('fly:203.0.113.10');
    expect(clientB).toBe('fly:203.0.113.11');
  });

  it('ignores a spoofed Fly-Client-IP outside a Fly runtime', async () => {
    await expect(
      clientIpTrackerForEnv(
        {
          headers: { 'fly-client-ip': '203.0.113.99' },
          ip: '198.51.100.20',
        },
        {}
      )
    ).resolves.toBe('direct:198.51.100.20');
  });

  it('requires both Fly runtime identity variables before trusting the proxy header', async () => {
    await expect(
      clientIpTrackerForEnv(
        {
          headers: { 'fly-client-ip': '203.0.113.99' },
          ip: '198.51.100.20',
        },
        { FLY_APP_NAME: 'woof-api-ci' }
      )
    ).resolves.toBe('direct:198.51.100.20');
  });

  it('fails safe to the direct peer when the Fly header is missing or malformed', async () => {
    await expect(
      clientIpTrackerForEnv(
        {
          headers: { 'fly-client-ip': '203.0.113.10, 198.51.100.7' },
          ip: '172.16.0.10',
        },
        flyEnv
      )
    ).resolves.toBe('fly-fallback:172.16.0.10');

    await expect(
      clientIpTrackerForEnv(
        {
          headers: { 'fly-client-ip': ['203.0.113.10', '203.0.113.11'] },
          socket: { remoteAddress: '172.16.0.11' },
        },
        flyEnv
      )
    ).resolves.toBe('fly-fallback:172.16.0.11');
  });

  it('accepts valid IPv6 client addresses', () => {
    expect(flyClientIp({ headers: { 'fly-client-ip': '2001:db8::1' } })).toBe('2001:db8::1');
  });

  it('detects only complete Fly runtime identity', () => {
    expect(isTrustedFlyRuntime(flyEnv)).toBe(true);
    expect(isTrustedFlyRuntime({ FLY_APP_NAME: 'woof-api-ci' })).toBe(false);
    expect(isTrustedFlyRuntime({ FLY_MACHINE_ID: 'machine-ci' })).toBe(false);
  });
});
