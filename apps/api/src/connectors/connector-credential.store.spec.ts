import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorCryptoService } from './connector-crypto.service';

const userId = '11111111-1111-4111-8111-111111111111';
const key = Buffer.alloc(32, 9).toString('base64');

function harness() {
  const integrationToken = {
    findMany: jest.fn().mockResolvedValue([]),
    count: jest.fn().mockResolvedValue(0),
    upsert: jest.fn().mockResolvedValue({ id: 'token-1' }),
    findUnique: jest.fn().mockResolvedValue(null),
    deleteMany: jest.fn().mockResolvedValue({ count: 1 }),
  };
  const prisma = { integrationToken } as unknown as PrismaService;
  const crypto = new ConnectorCryptoService({
    get: jest.fn((name: string) => (name === 'CONNECTOR_CREDENTIALS_KEY' ? key : undefined)),
  } as unknown as ConfigService);
  return {
    integrationToken,
    store: new ConnectorCredentialStore(prisma, crypto),
  };
}

describe('ConnectorCredentialStore', () => {
  it('stores only an encrypted envelope and normalized scopes', async () => {
    const { store, integrationToken } = harness();

    await store.put(
      userId,
      'FI',
      { accessToken: 'access-secret', refreshToken: 'refresh-secret' },
      ['activity', 'profile', 'activity'],
      new Date('2026-09-01T00:00:00.000Z'),
    );

    const args = integrationToken.upsert.mock.calls[0]?.[0];
    const stored = JSON.stringify(args.create.data);
    expect(stored).not.toContain('access-secret');
    expect(stored).not.toContain('refresh-secret');
    expect(args.create.provider).toBe('dogos_connector:FI');
    expect(args.create.scopes).toEqual(['activity', 'profile']);
  });

  it('decrypts a stored envelope only in its original user/provider context', async () => {
    const { store, integrationToken } = harness();
    await store.put(userId, 'TRACTIVE', { accessToken: 'secret' }, ['activity'], null);
    const envelope = integrationToken.upsert.mock.calls[0]?.[0]?.create?.data;
    integrationToken.findUnique.mockResolvedValue({
      data: envelope,
      scopes: ['activity'],
      expiresAt: null,
    });

    await expect(store.get(userId, 'TRACTIVE')).resolves.toEqual({
      credentials: { accessToken: 'secret' },
      scopes: ['activity'],
      expiresAt: null,
    });
  });

  it('lists connection metadata without decrypting credential payloads', async () => {
    const { store, integrationToken } = harness();
    integrationToken.findMany.mockResolvedValue([
      {
        provider: 'dogos_connector:FI',
        scopes: ['activity'],
        expiresAt: new Date('2026-09-01T00:00:00.000Z'),
        createdAt: new Date('2026-08-22T00:00:00.000Z'),
      },
    ]);

    await expect(store.listMetadata(userId)).resolves.toEqual([
      {
        provider: 'FI',
        scopes: ['activity'],
        expiresAt: '2026-09-01T00:00:00.000Z',
        createdAt: '2026-08-22T00:00:00.000Z',
      },
    ]);
  });

  it('disconnects by deleting only the connector credential row', async () => {
    const { store, integrationToken } = harness();

    await store.remove(userId, 'CHEWY');

    expect(integrationToken.deleteMany).toHaveBeenCalledWith({
      where: { userId, provider: 'dogos_connector:CHEWY' },
    });
  });
});
