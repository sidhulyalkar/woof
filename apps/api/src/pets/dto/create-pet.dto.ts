import { ApiProperty, ApiPropertyOptional, PartialType } from '@nestjs/swagger';
import { IsArray, IsISO8601, IsIn, IsNotEmpty, IsObject, IsOptional, IsString, IsUrl, ValidateIf } from 'class-validator';

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
    description: 'Temperament questionnaire scores or a list of descriptive traits.',
    example: { friendly: 5, energetic: 4, confident: 3 },
  })
  @IsOptional()
  @ValidateIf((_object, value) => value !== undefined)
  temperament?: Record<string, string | number | boolean> | string[];

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

export class UpdatePetDto extends PartialType(CreatePetDto) {}
