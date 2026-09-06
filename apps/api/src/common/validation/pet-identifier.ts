import { Matches, type ValidationOptions } from 'class-validator';

export const PET_IDENTIFIER_PATTERN =
  /^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|pet_[0-9a-f]{32})$/i;

export const PET_IDENTIFIER_DESCRIPTION =
  'a UUID-shaped legacy pet ID or replay-safe pet_<32hex> identifier';

export function IsPetIdentifier(options?: ValidationOptions): PropertyDecorator {
  return Matches(PET_IDENTIFIER_PATTERN, {
    message: `petId must be ${PET_IDENTIFIER_DESCRIPTION}`,
    ...options,
  });
}
