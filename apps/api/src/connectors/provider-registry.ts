import { BadRequestException } from '@nestjs/common';
import {
  CONNECTOR_PROVIDERS,
  type ConnectorProvider,
  type ConnectorProviderDefinition,
} from './connectors.types';

export const CONNECTOR_PROVIDER_REGISTRY: Record<ConnectorProvider, ConnectorProviderDefinition> = {
  FI: {
    provider: 'FI',
    label: 'Fi',
    domain: 'WEARABLE',
    availability: 'PARTNER_REQUIRED',
    capabilities: ['DAILY_ACTIVITY', 'DEVICE_STATUS'],
    preciseLocationEnabled: false,
    canonicalPetMutationAllowed: false,
    autonomousPurchaseAllowed: false,
    notes:
      'Wearable normalization is implemented, but dogOS does not claim a supported public Fi OAuth/API contract without verified partner documentation.',
  },
  TRACTIVE: {
    provider: 'TRACTIVE',
    label: 'Tractive',
    domain: 'WEARABLE',
    availability: 'PARTNER_REQUIRED',
    capabilities: ['DAILY_ACTIVITY', 'DEVICE_STATUS'],
    preciseLocationEnabled: false,
    canonicalPetMutationAllowed: false,
    autonomousPurchaseAllowed: false,
    notes:
      'Wearable normalization is implemented, but dogOS does not claim a supported public Tractive OAuth/API contract without verified partner documentation.',
  },
  VET_PARTNER: {
    provider: 'VET_PARTNER',
    label: 'Veterinary partner',
    domain: 'VET',
    availability: 'PARTNER_REQUIRED',
    capabilities: [
      'APPOINTMENT_IMPORT',
      'VACCINATION_IMPORT',
      'MEDICATION_REFERENCE',
      'DOCUMENT_REFERENCE',
    ],
    preciseLocationEnabled: false,
    canonicalPetMutationAllowed: false,
    autonomousPurchaseAllowed: false,
    notes:
      'Veterinary transport is provider-specific. Records must retain source provenance and remain references/imported observations unless an explicit domain workflow promotes them.',
  },
  CHEWY: {
    provider: 'CHEWY',
    label: 'Chewy',
    domain: 'RETAIL',
    availability: 'PARTNER_REQUIRED',
    capabilities: ['CATALOG_REFERENCE', 'USER_APPROVED_HANDOFF'],
    preciseLocationEnabled: false,
    canonicalPetMutationAllowed: false,
    autonomousPurchaseAllowed: false,
    notes:
      'Retail integration is approval-first. dogOS does not autonomously add to cart, place orders, or charge payment methods.',
  },
  PETCO: {
    provider: 'PETCO',
    label: 'Petco',
    domain: 'RETAIL',
    availability: 'PARTNER_REQUIRED',
    capabilities: ['CATALOG_REFERENCE', 'USER_APPROVED_HANDOFF'],
    preciseLocationEnabled: false,
    canonicalPetMutationAllowed: false,
    autonomousPurchaseAllowed: false,
    notes:
      'Retail integration is approval-first. dogOS does not autonomously add to cart, place orders, or charge payment methods.',
  },
};

export function parseConnectorProvider(value: string): ConnectorProvider {
  const provider = value.toUpperCase() as ConnectorProvider;
  if (!CONNECTOR_PROVIDERS.includes(provider)) {
    throw new BadRequestException('Unsupported dogOS connector provider');
  }
  return provider;
}
