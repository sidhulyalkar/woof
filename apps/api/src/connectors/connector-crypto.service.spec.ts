import { ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ConnectorCryptoService } from './connector-crypto.service';

const key = Buffer.alloc(32, 7).toString('base64');

function service(configuredKey?: string) {
  return new ConnectorCryptoService({
    get: jest.fn((name: string) =>
      name === 'CONNECTOR_CREDENTIALS_KEY' ? configuredKey : undefined,
    ),
  } as unknown as ConfigService);
}

describe('ConnectorCryptoService', () => {
  it('round-trips connector credentials without exposing token plaintext', () => {
    const crypto = service(key);
    const envelope = crypto.encrypt(
      { accessToken: 'secret-access-token', refreshToken: 'secret-refresh-token' },
      'user-1:FI',
    );

    expect(JSON.stringify(envelope)).not.toContain('secret-access-token');
    expect(JSON.stringify(envelope)).not.toContain('secret-refresh-token');
    expect(crypto.decrypt(envelope, 'user-1:FI')).toEqual({
      accessToken: 'secret-access-token',
      refreshToken: 'secret-refresh-token',
    });
  });

  it('binds ciphertext to its user/provider context with authenticated AAD', () => {
    const crypto = service(key);
    const envelope = crypto.encrypt({ accessToken: 'secret' }, 'user-1:FI');

    expect(() => crypto.decrypt(envelope, 'user-2:FI')).toThrow(ServiceUnavailableException);
    expect(() => crypto.decrypt(envelope, 'user-1:TRACTIVE')).toThrow(
      ServiceUnavailableException,
    );
  });

  it('fails closed when credential encryption is not configured', () => {
    const crypto = service();

    expect(crypto.isConfigured()).toBe(false);
    expect(() => crypto.encrypt({ accessToken: 'secret' }, 'user-1:FI')).toThrow(
      ServiceUnavailableException,
    );
  });

  it('rejects a deployment key that does not decode to 32 bytes', () => {
    expect(() => service(Buffer.alloc(16).toString('base64'))).toThrow(
      'CONNECTOR_CREDENTIALS_KEY must decode to exactly 32 bytes',
    );
  });
});
