import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { createCipheriv, createDecipheriv, randomBytes } from 'crypto';
import type { ConnectorCredentialEnvelope } from './connectors.types';

const KEY_BYTES = 32;
const IV_BYTES = 12;

@Injectable()
export class ConnectorCryptoService {
  private readonly key: Buffer | null;

  constructor(private readonly config: ConfigService) {
    const encoded = this.config.get<string>('CONNECTOR_CREDENTIALS_KEY');
    this.key = encoded ? this.parseKey(encoded) : null;
  }

  isConfigured() {
    return this.key !== null;
  }

  encrypt(value: Record<string, unknown>, context: string): ConnectorCredentialEnvelope {
    const key = this.requireKey();
    const iv = randomBytes(IV_BYTES);
    const cipher = createCipheriv('aes-256-gcm', key, iv);
    cipher.setAAD(Buffer.from(context, 'utf8'));
    const ciphertext = Buffer.concat([
      cipher.update(JSON.stringify(value), 'utf8'),
      cipher.final(),
    ]);
    const tag = cipher.getAuthTag();

    return {
      v: 1,
      alg: 'A256GCM',
      iv: iv.toString('base64url'),
      tag: tag.toString('base64url'),
      ciphertext: ciphertext.toString('base64url'),
    };
  }

  decrypt(envelope: ConnectorCredentialEnvelope, context: string): Record<string, unknown> {
    const key = this.requireKey();
    if (envelope.v !== 1 || envelope.alg !== 'A256GCM') {
      throw new ServiceUnavailableException('Unsupported connector credential envelope');
    }

    try {
      const decipher = createDecipheriv('aes-256-gcm', key, Buffer.from(envelope.iv, 'base64url'));
      decipher.setAAD(Buffer.from(context, 'utf8'));
      decipher.setAuthTag(Buffer.from(envelope.tag, 'base64url'));
      const plaintext = Buffer.concat([
        decipher.update(Buffer.from(envelope.ciphertext, 'base64url')),
        decipher.final(),
      ]).toString('utf8');
      const parsed: unknown = JSON.parse(plaintext);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('credential payload is not an object');
      }
      return parsed as Record<string, unknown>;
    } catch {
      throw new ServiceUnavailableException('Connector credentials could not be decrypted');
    }
  }

  private requireKey() {
    if (!this.key) {
      throw new ServiceUnavailableException(
        'Connector credential encryption is not configured for this environment'
      );
    }
    return this.key;
  }

  private parseKey(encoded: string) {
    let key: Buffer;
    try {
      key = Buffer.from(encoded, 'base64');
    } catch {
      throw new Error('CONNECTOR_CREDENTIALS_KEY must be a base64-encoded 32-byte key');
    }
    if (key.length !== KEY_BYTES) {
      throw new Error('CONNECTOR_CREDENTIALS_KEY must decode to exactly 32 bytes');
    }
    return key;
  }
}
