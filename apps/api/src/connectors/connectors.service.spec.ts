import { ConflictException, UnauthorizedException } from '@nestjs/common';
import { AutopilotService } from '../autopilot/autopilot.service';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorOperationalStore } from './connector-operational.store';
import { ConnectorsService } from './connectors.service';

const userId = '11111111-1111-4111-8111-111111111111';
const petId = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
const connection = {
  id: '22222222-2222-4222-8222-222222222222',
  provider: 'TRACTIVE' as const,
  status: 'CONNECTED' as const,
  externalAccountId: 'account-1',
  displayLabel: null,
  grantedScopes: ['activity'],
  connectedAt: '2026-08-22T00:00:00.000Z',
  lastSyncAt: null,
  revokedAt: null,
  createdAt: '2026-08-22T00:00:00.000Z',
  updatedAt: '2026-08-22T00:00:00.000Z',
};

function harness() {
  const credentials = {
    encryptionConfigured: jest.fn().mockReturnValue(true),
    state: jest.fn().mockResolvedValue('MISSING'),
    put: jest.fn().mockResolvedValue(undefined),
    remove: jest.fn().mockResolvedValue(undefined),
  };
  const operational = {
    listConnections: jest.fn().mockResolvedValue([]),
    getConnection: jest.fn().mockResolvedValue(null),
    markConnected: jest.fn().mockResolvedValue(connection),
    markReauthRequired: jest.fn().mockResolvedValue(undefined),
    bindPetIdentity: jest.fn().mockResolvedValue(null),
    getPetIdentity: jest.fn().mockResolvedValue(null),
    getImportReceipt: jest.fn().mockResolvedValue(null),
    recordImportReceipt: jest.fn().mockImplementation(async (input) => ({
      id: 'receipt-1',
      ...input,
      importedAt: '2026-08-22T08:00:00.000Z',
    })),
    markLocallyRevoked: jest.fn().mockResolvedValue('revocation-1'),
  };
  const autopilot = {
    ingestProviderObservation: jest.fn().mockResolvedValue({
      careEventId: 'care-event-1',
      duplicate: false,
      bondXp: 0,
      observation: {
        provider: 'TRACTIVE',
        externalEventId: 'day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: new Date('2026-08-22T08:00:00.000Z'),
        metrics: { activityMinutes: 51 },
      },
      signal: null,
    }),
  };
  return {
    credentials,
    operational,
    autopilot,
    service: new ConnectorsService(
      credentials as unknown as ConnectorCredentialStore,
      operational as unknown as ConnectorOperationalStore,
      autopilot as unknown as AutopilotService,
    ),
  };
}

describe('ConnectorsService', () => {
  it('reports partner-gated providers without fake connected states', async () => {
    const { service } = harness();

    const dashboard = await service.getDashboard(userId);

    expect(dashboard.providers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          provider: 'FI',
          availability: 'PARTNER_REQUIRED',
          preciseLocationEnabled: false,
        }),
        expect.objectContaining({
          provider: 'CHEWY',
          availability: 'PARTNER_REQUIRED',
          autonomousPurchaseAllowed: false,
        }),
      ]),
    );
    expect(dashboard.boundaries).toEqual(
      expect.objectContaining({
        undocumentedOAuthAllowed: false,
        browserProviderImpersonationAllowed: false,
        canonicalPetMutationAllowed: false,
        autonomousRetailPurchaseAllowed: false,
        rawProviderPayloadStored: false,
      }),
    );
  });

  it('refuses to invent OAuth authorization URLs for partner-gated providers', () => {
    const { service } = harness();

    expect(() => service.startOAuth('fi')).toThrow(ConflictException);
    expect(() => service.startOAuth('chewy')).toThrow(ConflictException);
  });

  it('shows connected only when operational state and credential authentication both pass', async () => {
    const { service, operational, credentials } = harness();
    operational.listConnections.mockResolvedValue([connection]);
    credentials.state.mockResolvedValue('USABLE');

    const dashboard = await service.getDashboard(userId);
    const tractive = dashboard.providers.find((provider) => provider.provider === 'TRACTIVE');

    expect(tractive).toEqual(
      expect.objectContaining({
        availability: 'CONNECTED',
        credentialState: 'USABLE',
        connection: expect.objectContaining({ id: connection.id }),
      }),
    );
  });

  it('degrades expired connected credentials to reauthorization', async () => {
    const { service, operational, credentials } = harness();
    operational.listConnections.mockResolvedValue([connection]);
    credentials.state.mockResolvedValue('EXPIRED');

    const dashboard = await service.getDashboard(userId);
    const tractive = dashboard.providers.find((provider) => provider.provider === 'TRACTIVE');

    expect(tractive).toEqual(
      expect.objectContaining({
        availability: 'REAUTH_REQUIRED',
        credentialState: 'EXPIRED',
      }),
    );
    expect(operational.markReauthRequired).toHaveBeenCalledWith(userId, 'TRACTIVE');
  });

  it('registers credentials before declaring a transport-verified connection', async () => {
    const { service, credentials, operational } = harness();

    await service.registerVerifiedConnectionFromTransport({
      userId,
      provider: 'FI',
      externalAccountId: 'fi-account-1',
      displayLabel: 'Collar',
      grantedScopes: ['activity'],
      credentials: { accessToken: 'secret' },
      expiresAt: null,
    });

    expect(credentials.put).toHaveBeenCalled();
    expect(operational.markConnected).toHaveBeenCalledWith(
      expect.objectContaining({ userId, provider: 'FI', externalAccountId: 'fi-account-1' }),
    );
    expect(credentials.put.mock.invocationCallOrder[0]).toBeLessThan(
      operational.markConnected.mock.invocationCallOrder[0]!,
    );
  });

  it('requires a connected account, usable credential, and mapped provider pet before import', async () => {
    const { service, operational, credentials } = harness();

    await expect(
      service.ingestWearableFromTransport(userId, 'FI', {
        externalPetId: 'fi-pet-1',
        externalObjectId: 'fi-day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { activityMinutes: 45 },
      }),
    ).rejects.toBeInstanceOf(UnauthorizedException);

    operational.getConnection.mockResolvedValue({ ...connection, provider: 'FI' });
    credentials.state.mockResolvedValue('USABLE');
    await expect(
      service.ingestWearableFromTransport(userId, 'FI', {
        externalPetId: 'fi-pet-1',
        externalObjectId: 'fi-day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { activityMinutes: 45 },
      }),
    ).rejects.toBeInstanceOf(UnauthorizedException);
  });

  it('reuses Autopilot source truth and records hash-only provenance after verified import', async () => {
    const { service, operational, credentials, autopilot } = harness();
    operational.getConnection.mockResolvedValue(connection);
    credentials.state.mockResolvedValue('USABLE');
    operational.getPetIdentity.mockResolvedValue({
      connectionId: connection.id,
      petId,
      externalPetId: 'tractive-pet-1',
      externalPetLabel: 'Scout',
      verifiedAt: '2026-08-22T00:00:00.000Z',
    });

    await expect(
      service.ingestWearableFromTransport(userId, 'tractive', {
        externalPetId: 'tractive-pet-1',
        externalObjectId: 'day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { activityMinutes: 51 },
      }),
    ).resolves.toEqual(
      expect.objectContaining({ careEventId: 'care-event-1', bondXp: 0 }),
    );

    expect(autopilot.ingestProviderObservation).toHaveBeenCalledWith(
      userId,
      'TRACTIVE',
      expect.objectContaining({ petId, externalEventId: 'day-1' }),
    );
    expect(operational.recordImportReceipt).toHaveBeenCalledWith(
      expect.objectContaining({
        connectionId: connection.id,
        disposition: 'IMPORTED',
        canonicalRefType: 'CARE_EVENT',
        canonicalRefId: 'care-event-1',
        payloadHash: expect.stringMatching(/^[0-9a-f]{64}$/),
      }),
    );
  });

  it('repairs a missing receipt from canonical truth and rejects an altered replay payload', async () => {
    const { service, operational, credentials, autopilot } = harness();
    operational.getConnection.mockResolvedValue(connection);
    credentials.state.mockResolvedValue('USABLE');
    operational.getPetIdentity.mockResolvedValue({
      connectionId: connection.id,
      petId,
      externalPetId: 'tractive-pet-1',
      externalPetLabel: null,
      verifiedAt: '2026-08-22T00:00:00.000Z',
    });
    autopilot.ingestProviderObservation.mockResolvedValue({
      careEventId: 'care-event-1',
      duplicate: true,
      bondXp: 0,
      observation: {
        provider: 'TRACTIVE',
        externalEventId: 'day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: new Date('2026-08-22T08:00:00.000Z'),
        metrics: { activityMinutes: 51 },
      },
      signal: null,
    });

    await expect(
      service.ingestWearableFromTransport(userId, 'TRACTIVE', {
        externalPetId: 'tractive-pet-1',
        externalObjectId: 'day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { activityMinutes: 99 },
      }),
    ).rejects.toBeInstanceOf(ConflictException);

    expect(operational.recordImportReceipt).toHaveBeenCalledWith(
      expect.objectContaining({ canonicalRefId: 'care-event-1', disposition: 'IMPORTED' }),
    );
  });

  it('disconnects local credentials and records local-only revocation evidence', async () => {
    const { service, credentials, operational } = harness();
    operational.getConnection.mockResolvedValue({ ...connection, provider: 'PETCO' });

    await expect(service.disconnect(userId, 'PETCO')).resolves.toEqual({
      success: true,
      provider: 'PETCO',
      localCredentialsRemoved: true,
      localRevocationReceiptId: 'revocation-1',
      remoteRevocation: 'NOT_CONFIGURED',
    });
    expect(credentials.remove).toHaveBeenCalledWith(userId, 'PETCO');
    expect(operational.markLocallyRevoked).toHaveBeenCalledWith(userId, 'PETCO');
  });

  it('never routes retail connectors through wearable ingestion', async () => {
    const { service } = harness();

    await expect(
      service.ingestWearableFromTransport(userId, 'CHEWY', {
        externalPetId: 'retail-pet',
        externalObjectId: 'item-1',
        kind: 'DEVICE_STATUS',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { batteryPercent: 50 },
      }),
    ).rejects.toBeInstanceOf(ConflictException);
  });
});
