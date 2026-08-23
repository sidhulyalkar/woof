import { ConflictException, Injectable, UnauthorizedException } from '@nestjs/common';
import type { IngestTrackerObservationDto } from '../autopilot/dto/autopilot.dto';
import { AutopilotService } from '../autopilot/autopilot.service';
import { ConnectorCredentialStore } from './connector-credential.store';
import type { ConnectorProvider } from './connectors.types';
import { CONNECTOR_PROVIDER_REGISTRY, parseConnectorProvider } from './provider-registry';

@Injectable()
export class ConnectorsService {
  constructor(
    private readonly credentials: ConnectorCredentialStore,
    private readonly autopilot: AutopilotService,
  ) {}

  async getDashboard(userId: string) {
    const connected = new Map(
      (await this.credentials.listMetadata(userId)).map((metadata) => [metadata.provider, metadata]),
    );

    return {
      providers: Object.values(CONNECTOR_PROVIDER_REGISTRY).map((definition) => {
        const metadata = connected.get(definition.provider);
        return {
          ...definition,
          availability: metadata ? ('CONNECTED' as const) : definition.availability,
          connection: metadata ?? null,
        };
      }),
      credentialEncryptionConfigured: this.credentials.encryptionConfigured(),
      boundaries: {
        undocumentedOAuthAllowed: false,
        preciseLocationImportEnabled: false,
        canonicalPetMutationAllowed: false,
        autonomousRetailPurchaseAllowed: false,
        importedWearablesRewardEligible: false,
      },
    };
  }

  startOAuth(providerValue: string) {
    const provider = parseConnectorProvider(providerValue);
    const definition = CONNECTOR_PROVIDER_REGISTRY[provider];

    throw new ConflictException({
      code: 'partner_required',
      provider,
      message: `${definition.label} does not have a verified OAuth transport configured in this dogOS release.`,
    });
  }

  async disconnect(userId: string, providerValue: string) {
    const provider = parseConnectorProvider(providerValue);
    await this.credentials.remove(userId, provider);
    return {
      success: true,
      provider,
      localCredentialsRemoved: true,
      remoteRevocation: 'NOT_CONFIGURED' as const,
    };
  }

  async importWearableObservation(
    userId: string,
    providerValue: string,
    dto: IngestTrackerObservationDto,
  ) {
    const provider = parseConnectorProvider(providerValue);
    this.assertWearable(provider);
    if (!(await this.credentials.has(userId, provider))) {
      throw new UnauthorizedException('A verified connector credential is required before import');
    }

    return this.autopilot.ingestProviderObservation(userId, provider, dto);
  }

  private assertWearable(provider: ConnectorProvider) {
    if (CONNECTOR_PROVIDER_REGISTRY[provider].domain !== 'WEARABLE') {
      throw new ConflictException('This connector does not import wearable observations');
    }
  }
}
