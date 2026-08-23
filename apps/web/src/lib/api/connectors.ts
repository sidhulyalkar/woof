import { apiClient } from './client';

export type ConnectorProvider = 'FI' | 'TRACTIVE' | 'VET_PARTNER' | 'CHEWY' | 'PETCO';
export type ConnectorDomain = 'WEARABLE' | 'VET' | 'RETAIL';
export type ConnectorAvailability =
  'PARTNER_REQUIRED' | 'CONNECTED' | 'REAUTH_REQUIRED' | 'REVOKED';
export type ConnectorCredentialState = 'MISSING' | 'USABLE' | 'EXPIRED' | 'INVALID';

export type ConnectorConnection = {
  id: string;
  provider: ConnectorProvider;
  status: ConnectorAvailability;
  externalAccountId: string | null;
  displayLabel: string | null;
  grantedScopes: string[];
  connectedAt: string | null;
  lastSyncAt: string | null;
  revokedAt: string | null;
  createdAt: string;
  updatedAt: string;
};

export type ConnectorProviderState = {
  provider: ConnectorProvider;
  label: string;
  domain: ConnectorDomain;
  availability: ConnectorAvailability;
  capabilities: string[];
  preciseLocationEnabled: false;
  canonicalPetMutationAllowed: false;
  autonomousPurchaseAllowed: false;
  notes: string;
  connection: ConnectorConnection | null;
  credentialState: ConnectorCredentialState;
};

export type ConnectorsDashboard = {
  providers: ConnectorProviderState[];
  credentialEncryptionConfigured: boolean;
  boundaries: {
    undocumentedOAuthAllowed: false;
    browserProviderImpersonationAllowed: false;
    preciseLocationImportEnabled: false;
    canonicalPetMutationAllowed: false;
    autonomousRetailPurchaseAllowed: false;
    importedWearablesRewardEligible: false;
    rawProviderPayloadStored: false;
  };
};

export type DisconnectConnectorResult = {
  success: true;
  provider: ConnectorProvider;
  localCredentialsRemoved: true;
  localRevocationReceiptId: string | null;
  remoteRevocation: 'NOT_CONFIGURED';
};

export const connectorsApi = {
  getDashboard: () => apiClient.get<ConnectorsDashboard>('/connectors'),
  disconnect: (provider: ConnectorProvider) =>
    apiClient.delete<DisconnectConnectorResult>(`/connectors/${provider}`),
};
