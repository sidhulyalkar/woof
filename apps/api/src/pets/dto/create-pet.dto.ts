import { ApiProperty, ApiPropertyOptional, OmitType, PartialType } from '@nestjs/swagger';
import {
  IsArray,
  IsISO8601,
  IsIn,
  IsNotEmpty,
  IsOptional,
  IsString,
  IsUrl,
  MaxLength,
} from 'class-validator';

export class CreatePetDto {
  @ApiProperty({ example: 'Shasta' })
  @IsString()
  @IsNotEmpty()
  name!: string;

  @ApiProperty({ example: 'DOG' })
  @IsString()
  @IsNotEmpty()
  species!: string;

  @ApiPropertyOptional({ example: 'Siberian Husky' })
  @IsOptional()
  @IsString()
  breed?: string;

  @ApiPropertyOptional({ enum: ['MALE', 'FEMALE', 'UNKNOWN'] })
  @IsOptional()
  @IsIn(['MALE', 'FEMALE', 'UNKNOWN'])
  sex?: 'MALE' | 'FEMALE' | 'UNKNOWN';

  @ApiPropertyOptional({ example: '2022-04-12' })
  @IsOptional()
  @IsISO8601()
  birthdate?: string;

  @ApiPropertyOptional({
    description:
      'Optional replay-safe key for a minimal onboarding create. Exact retries return the same pet. Media and mutable profile JSON should be attached after creation.',
    example: 'first-adventure:01J8Y6N8WJY5DR1PSE2K1XQ4QS',
  })
  @IsOptional()
  @IsString()
  @MaxLength(128)
  creationKey?: string;

  @ApiPropertyOptional({
    description:
      'Current descriptive temperament traits. Scored questionnaires belong in separate feature/preference records.',
    example: ['Friendly', 'Energetic', 'Social'],
    type: [String],
  })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  temperament?: string[];

  @ApiPropertyOptional({
    description: 'Vaccination records supplied by the owner.',
    type: 'array',
  })
  @IsOptional()
  @IsArray()
  vaccinations?: Array<Record<string, unknown>>;

  @ApiPropertyOptional({ example: 'https://cdn.example.com/pets/shasta.jpg' })
  @IsOptional()
  @IsUrl({ require_tld: false })
  avatarUrl?: string;
}

export class UpdatePetDto extends PartialType(OmitType(CreatePetDto, ['creationKey'] as const)) {}
