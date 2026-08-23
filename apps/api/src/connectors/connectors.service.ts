import { createHash } from 'node:crypto';
import { ConflictException, Injectable, UnauthorizedException } from '@nestjs/common';
import type { IngestTrackerObservationDto } from '../autopilot/dto/autopilot.dto';
import { normalizeProviderObservation } from '../autopilot/provider-adapters';
import { AutopilotService } from '../autopilot/autopilot.service';
import type { NormalizedTrackerObservation } from '../autopilot/autopilot.types';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorOperationalStore } from './connector-operational.store';
import type {
  ConnectorProvider,
  VerifiedWearableTransportEvent,
} from './connectors.types';
import { CONNECTOR_PROVIDER_REGISTRY, parseConnectorProvider } from './provider-registry';

function hashObservation(observation: NormalizedTrackerObservation) {
  const metrics = Object.fromEntries(
    Object.entries(observation.metrics)
      .filter(([, value]) => value !== undefined)
      .sort(([left], [right]) => left.localeCompare(right)),
  );
  const canonical = JSON.stringify({
    provider: observation.provider,
    externalEventId: observation.externalEventId,
    kind: observation.kind,
    observedAt: observation.observedAt.toISOString(),
    metrics,
  });
  return createHash('sha256').update(canonical).digest('hex');
}

@Injectable()
export class ConnectorsService {
  constructor(
    private readonly credentials: ConnectorCredentialStore,
    private readonly operational: ConnectorOperationalStore,
    private readonly autopilot: AutopilotService,
  ) {}

  async getDashboard(userId: string) {
    const connections = new Map(
      (await this.operational.listConnections(userId)).map((connection) => [
        connection.provider,
        connection,
      ]),
    );

    const providers = await Promise.all(
      Object.values(CONNECTOR_PROVIDER_REGISTRY).map(async (definition) => {
        const connection = connections.get(definition.provider) ?? null;
        if (!connection) {
          return {
            ...definition,
            availability: definition.availability,
            connection: null,
            credentialState: 'MISSING' as const,
          };
        }

        if (connection.status === 'REVOKED') {
          return {
            ...definition,
            availability: 'REVOKED' as const,
            connection,
            credentialState: 'MISSING' as const,
          };
        }

        if (connection.status !== 'CONNECTED') {
          return {
            ...definition,
            availability: connection.status,
            connection,
            credentialState: 'MISSING' as const,
          };
        }

        const credentialState = await this.credentials.state(userId, definition.provider);
        if (credentialState !== 'USABLE') {
          await this.operational.markReauthRequired(userId, definition.provider);
          return {
            ...definition,
            availability: 'REAUTH_REQUIRED' as const,
            connection: { ...connection, status: 'REAUTH_REQUIRED' as const },
            credentialState,
          };
        }

        return {
          ...definition,
          availability: 'CONNECTED' as const,
          connection,
          credentialState,
        };
      }),
    );

    return {
      providers,
      credentialEncryptionConfigured: this.credentials.encryptionConfigured(),
      boundaries: {
        undocumentedOAuthAllowed: false,
        browserProviderImpersonationAllowed: false,
        preciseLocationImportEnabled: false,
        canonicalPetMutationAllowed: false,
        autonomousRetailPurchaseAllowed: false,
        importedWearablesRewardEligible: false,
        rawProviderPayloadStored: false,
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
    const connection = await this.operational.getConnection(userId, provider);
    await this.credentials.remove(userId, provider);
    const revocationReceiptId = connection
      ? await this.operational.markLocallyRevoked(userId, provider)
      : null;

    return {
      success: true,
      provider,
      localCredentialsRemoved: true,
      localRevocationReceiptId: revocationReceiptId,
      remoteRevocation: 'NOT_CONFIGURED' as const,
    };
  }

  /**
   * Internal transport seam. Future verified OAuth callbacks call this only after
   * validating the provider response. It is intentionally not exposed by the
   * public ConnectorsController.
   */
  async registerVerifiedConnectionFromTransport(input: {
    userId: string;
    provider: ConnectorProvider;
    externalAccountId: string | null;
    displayLabel: string | null;
    grantedScopes: string[];
    credentials: Record<string, unknown>;
    expiresAt: Date | null;
  }) {
    await this.credentials.put(
      input.userId,
      input.provider,
      input.credentials,
      input.grantedScopes,
      input.expiresAt,
    );
    return this.operational.markConnected({
      userId: input.userId,
      provider: input.provider,
      externalAccountId: input.externalAccountId,
      displayLabel: input.displayLabel,
      grantedScopes: input.grantedScopes,
    });
  }

  /** Map a provider-owned pet identity to an owned dogOS pet after provider discovery. */
  async bindPetIdentityFromTransport(input: {
    userId: string;
    provider: ConnectorProvider;
    petId: string;
    externalPetId: string;
    externalPetLabel?: string | null;
  }) {
    const identity = await this.operational.bindPetIdentity(input);
    if (!identity) {
      throw new ConflictException('A connected provider and owned pet are required for identity mapping');
    }
    return identity;
  }

  /**
   * Internal provider-ingestion seam. Browser callers cannot supply provider data.
   * The external pet identity is resolved through connector metadata before the
   * normalized summary is delegated to Autopilot/CareEvent source truth.
   */
  async ingestWearableFromTransport(
    userId: string,
    providerValue: string,
    event: VerifiedWearableTransportEvent,
  ) {
    const provider = parseConnectorProvider(providerValue);
    this.assertWearable(provider);

    const connection = await this.operational.getConnection(userId, provider);
    if (!connection || connection.status !== 'CONNECTED') {
      throw new UnauthorizedException('A connected provider account is required before import');
    }

    const credentialState = await this.credentials.state(userId, provider);
    if (credentialState !== 'USABLE') {
      await this.operational.markReauthRequired(userId, provider);
      throw new UnauthorizedException('The connector credential requires reauthorization');
    }

    const identity = await this.operational.getPetIdentity(userId, provider, event.externalPetId);
    if (!identity) {
      throw new UnauthorizedException('The provider pet is not mapped to an owned dogOS pet');
    }

    const dto: IngestTrackerObservationDto = {
      petId: identity.petId,
      externalEventId: event.externalObjectId,
      kind: event.kind,
      observedAt: event.observedAt,
      payload: event.payload,
    };
    const requestedObservation = normalizeProviderObservation(provider, dto);
    const requestedHash = hashObservation(requestedObservation);
    const resourceType = `WEARABLE_${event.kind}`;

    const existing = await this.operational.getImportReceipt(
      identity.connectionId,
      resourceType,
      event.externalObjectId,
    );
    if (existing) {
      if (existing.payloadHash !== requestedHash) {
        throw new ConflictException({
          code: 'external_object_changed_after_import',
          provider,
          externalObjectId: event.externalObjectId,
        });
      }
      return { duplicate: true, receipt: existing };
    }

    const result = await this.autopilot.ingestProviderObservation(userId, provider, dto);
    const canonicalHash = hashObservation(result.observation);
    const receipt = await this.operational.recordImportReceipt({
      connectionId: identity.connectionId,
      resourceType,
      externalObjectId: event.externalObjectId,
      payloadHash: canonicalHash,
      disposition: 'IMPORTED',
      canonicalRefType: 'CARE_EVENT',
      canonicalRefId: result.careEventId,
      occurredAt: result.observation.observedAt,
    });

    if (!receipt) throw new ConflictException('Connector import receipt could not be persisted');
    if (receipt.payloadHash !== requestedHash || canonicalHash !== requestedHash) {
      throw new ConflictException({
        code: 'external_object_changed_after_import',
        provider,
        externalObjectId: event.externalObjectId,
      });
    }

    return {
      duplicate: result.duplicate,
      careEventId: result.careEventId,
      bondXp: result.bondXp,
      receipt,
    };
  }

  private assertWearable(provider: ConnectorProvider) {
    if (CONNECTOR_PROVIDER_REGISTRY[provider].domain !== 'WEARABLE') {
      throw new ConflictException('This connector does not import wearable observations');
    }
  }
}
