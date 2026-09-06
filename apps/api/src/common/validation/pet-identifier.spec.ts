import 'reflect-metadata';
import { validate } from 'class-validator';
import { IsPetIdentifier, PET_IDENTIFIER_PATTERN } from './pet-identifier';

class SinglePetIdentifierDto {
  @IsPetIdentifier()
  petId!: string;
}

class PetIdentifierListDto {
  @IsPetIdentifier({ each: true })
  petIds!: string[];
}

describe('canonical pet identifier validation', () => {
  it.each([
    '6f78f3b5-06b6-4a19-8caf-90a3a1c879be',
    '00000000-0000-0000-0000-000000000000',
    'pet_0123456789abcdef0123456789abcdef',
    'PET_0123456789ABCDEF0123456789ABCDEF',
  ])('accepts persisted canonical pet identifier %s', async (petId) => {
    const dto = new SinglePetIdentifierDto();
    dto.petId = petId;
    await expect(validate(dto)).resolves.toHaveLength(0);
    expect(PET_IDENTIFIER_PATTERN.test(petId)).toBe(true);
  });

  it.each([
    '',
    'pet_0123456789abcdef',
    'pet_0123456789abcdef0123456789abcdef00',
    'pet_not-hexadecimal-identifier',
    'user_0123456789abcdef0123456789abcdef',
    'not-a-pet-id',
  ])('rejects non-canonical pet identifier %s', async (petId) => {
    const dto = new SinglePetIdentifierDto();
    dto.petId = petId;
    await expect(validate(dto)).resolves.not.toHaveLength(0);
  });

  it('validates every pet identifier in bounded arrays', async () => {
    const valid = new PetIdentifierListDto();
    valid.petIds = ['6f78f3b5-06b6-4a19-8caf-90a3a1c879be', 'pet_0123456789abcdef0123456789abcdef'];
    await expect(validate(valid)).resolves.toHaveLength(0);

    const invalid = new PetIdentifierListDto();
    invalid.petIds = [valid.petIds[0], 'pet_wrong'];
    await expect(validate(invalid)).resolves.not.toHaveLength(0);
  });
});
