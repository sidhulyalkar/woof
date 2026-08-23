import { ConflictException, UnauthorizedException } from '@nestjs/common';
import { AutopilotService } from '../autopilot/autopilot.service';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorsService } from './connectors.service';

const userId = '11111111-1111-4111-8111-111111111111';

function harness() {
  const credentials = {
    listMetadata: jest.fn().mockResolvedValue([]),
    encryptionConfigured: jest.fn().mockReturnValue(true),
    remove: jest.fn().mockResolvedValue(undefined),
    has: jest.fn().mockResolvedValue(false),
  };
  const autopilot = {
    ingestProviderObservation: jest.fn().mockResolvedValue({ duplicate: false }),
  };
  return {
    credentials,
    autopilot,
    service: new ConnectorsService(
      credentials as unknown as ConnectorCredentialStore,
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
        canonicalPetMutationAllowed: false,
        autonomousRetailPurchaseAllowed: false,
      }),
    );
  });

  it('refuses to invent OAuth authorization URLs for partner-gated providers', () => {
    const { service } = harness();

    expect(() => service.startOAuth('fi')).toThrow(ConflictException);
    expect(() => service.startOAuth('chewy')).toThrow(ConflictException);
  });

  it('shows connected only when an encrypted credential row actually exists', async () => {
    const { service, credentials } = harness();
    credentials.listMetadata.mockResolvedValue([
      {
        provider: 'TRACTIVE',
        scopes: ['activity'],
        expiresAt: null,
        createdAt: '2026-08-22T00:00:00.000Z',
      },
    ]);

    const dashboard = await service.getDashboard(userId);
    const tractive = dashboard.providers.find((provider) => provider.provider === 'TRACTIVE');

    expect(tractive).toEqual(
      expect.objectContaining({
        availability: 'CONNECTED',
        connection: expect.objectContaining({ scopes: ['activity'] }),
      }),
    );
  });

  it('requires a verified wearable credential before routing an import to Autopilot', async () => {
    const { service } = harness();

    await expect(
      service.importWearableObservation(userId, 'FI', {
        petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
        externalEventId: 'fi-day-1',
        kind: 'DAILY_ACTIVITY',
        observedAt: '2026-08-22T08:00:00.000Z',
        payload: { activityMinutes: 45 },
      }),
    ).rejects.toBeInstanceOf(UnauthorizedException);
  });

  it('reuses the Autopilot normalization and zero-reward path after connection verification', async () => {
    const { service, credentials, autopilot } = harness();
    credentials.has.mockResolvedValue(true);
    const dto = {
      petId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
      externalEventId: 'tractive-day-1',
      kind: 'DAILY_ACTIVITY' as const,
      observedAt: '2026-08-22T08:00:00.000Z',
      payload: { activityMinutes: 51 },
    };

    await service.importWearableObservation(userId, 'tractive', dto);

    expect(autopilot.ingestProviderObservation).toHaveBeenCalledWith(
      userId,
      'TRACTIVE',
      dto,
    );
  });

  it('disconnects local connector credentials without claiming remote revocation', async () => {
    const { service, credentials } = harness();

    await expect(service.disconnect(userId, 'PETCO')).resolves.toEqual({
      success: true,
      provider: 'PETCO',
      localCredentialsRemoved: true,
      remoteRevocation: 'NOT_CONFIGURED',
    });
    expect(credentials.remove).toHaveBeenCalledWith(userId, 'PETCO');
  });
});
