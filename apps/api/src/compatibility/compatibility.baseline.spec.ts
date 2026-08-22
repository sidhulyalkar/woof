import type { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CompatibilityService } from './compatibility.service';

type PetFixture = {
  id: string;
  name: string;
  species: string;
  breed: string | null;
  birthdate: Date | null;
  temperament: Prisma.JsonValue | null;
};

describe('CompatibilityService deterministic baseline', () => {
  const service = new CompatibilityService({} as PrismaService);

  const basePet: PetFixture = {
    id: 'pet-a',
    name: 'Shasta',
    species: 'DOG',
    breed: 'Siberian Husky',
    birthdate: new Date('2022-01-01'),
    temperament: ['Friendly', 'Energetic', 'Social'],
  };

  const score = (a: PetFixture, b: PetFixture) => service['scoreBaseline'](a, b);

  it('returns exactly the same result for the same inputs', () => {
    const other: PetFixture = {
      ...basePet,
      id: 'pet-b',
      name: 'Nova',
      birthdate: new Date('2022-08-01'),
    };

    expect(score(basePet, other)).toEqual(score(basePet, other));
  });

  it('ranks strong temperament overlap above a disjoint profile', () => {
    const aligned: PetFixture = {
      ...basePet,
      id: 'pet-b',
      name: 'Nova',
      temperament: ['Friendly', 'Energetic', 'Social'],
    };
    const disjoint: PetFixture = {
      ...basePet,
      id: 'pet-c',
      name: 'Milo',
      temperament: ['Shy', 'Independent', 'Calm'],
    };

    expect(score(basePet, aligned).score).toBeGreaterThan(score(basePet, disjoint).score);
  });

  it('uses breed as a weak supporting signal rather than a hard gate', () => {
    const sameBreed: PetFixture = {
      ...basePet,
      id: 'pet-b',
      name: 'Nova',
      breed: 'Siberian Husky',
    };
    const differentBreed: PetFixture = {
      ...basePet,
      id: 'pet-c',
      name: 'Milo',
      breed: 'Labrador Retriever',
    };

    const sameBreedScore = score(basePet, sameBreed).score;
    const differentBreedScore = score(basePet, differentBreed).score;

    expect(sameBreedScore).toBeGreaterThan(differentBreedScore);
    expect(sameBreedScore - differentBreedScore).toBeLessThan(0.1);
  });

  it('lowers confidence when profile signals are missing', () => {
    const complete: PetFixture = {
      ...basePet,
      id: 'pet-b',
      name: 'Nova',
    };
    const sparse: PetFixture = {
      id: 'pet-c',
      name: 'Milo',
      species: 'DOG',
      breed: null,
      birthdate: null,
      temperament: null,
    };

    expect(score(basePet, complete).confidence).toBeGreaterThan(score(basePet, sparse).confidence);
  });

  it('always returns a bounded score and confidence', () => {
    const other: PetFixture = {
      id: 'pet-b',
      name: 'Nova',
      species: 'CAT',
      breed: null,
      birthdate: null,
      temperament: null,
    };

    const result = score(basePet, other);
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});
