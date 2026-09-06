import { Type } from 'class-transformer';
import {
  IsIn,
  IsISO8601,
  IsOptional,
  IsString,
  Matches,
  MaxLength,
  ValidateNested,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';
import { DAILY_SIGNAL_CHOICES, type DailySignalChoice } from '../daily-signals.types';

const HOUSEHOLD_IDENTIFIER_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class DailySignalsAnswersDto {
  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  appetite?: DailySignalChoice;

  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  energy?: DailySignalChoice;

  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  bathroomRoutine?: DailySignalChoice;

  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  mobilityComfort?: DailySignalChoice;

  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  engagementSocialComfort?: DailySignalChoice;

  @IsOptional()
  @IsIn(DAILY_SIGNAL_CHOICES)
  sleepRest?: DailySignalChoice;
}

export class CreateDailySignalsDto {
  // Personal household IDs predate strict RFC UUID version/variant bits. Keep
  // accepting the canonical stored 8-4-4-4-12 hexadecimal identifier without
  // rewriting an existing household into a different identity.
  @Matches(HOUSEHOLD_IDENTIFIER_PATTERN, {
    message: 'householdId must be a UUID-shaped identifier',
  })
  householdId!: string;

  @IsPetIdentifier()
  petId!: string;

  @IsOptional()
  @IsISO8601({ strict: true })
  observedAt?: string;

  @ValidateNested()
  @Type(() => DailySignalsAnswersDto)
  signals!: DailySignalsAnswersDto;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  note?: string;
}
