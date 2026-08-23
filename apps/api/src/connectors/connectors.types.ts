export const CONNECTOR_PROVIDERS = ['FI', 'TRACTIVE', 'VET_PARTNER', 'CHEWY', 'PETCO'] as const;
export type ConnectorProvider = (typeof CONNECTOR_PROVIDERS)[number];

export type ConnectorDomain = 'WEARABLE' | 'VET' | 'RETAIL';
export type ConnectorAvailability = 'PARTNER_REQUIRED' | 'CONNECTED';

export type ConnectorCapability =
  | 'DAILY_ACTIVITY'
  | 'DEVICE_STATUS'
  | 'APPOINTMENT_IMPORT'
  | 'VACCINATION_IMPORT'
  | 'MEDICATION_REFERENCE'
  | 'DOCUMENT_REFERENCE'
  | 'CATALOG_REFERENCE'
  | 'USER_APPROVED_HANDOFF';

export type ConnectorProviderDefinition = {
  provider: ConnectorProvider;
  label: string;
  domain: ConnectorDomain;
  availability: 'PARTNER_REQUIRED';
  capabilities: ConnectorCapability[];
  preciseLocationEnabled: false;
  canonicalPetMutationAllowed: false;
  autonomousPurchaseAllowed: false;
  notes: string;
};

export type ConnectorCredentialEnvelope = {
  v: 1;
  alg: 'A256GCM';
  iv: string;
  tag: string;
  ciphertext: string;
};
