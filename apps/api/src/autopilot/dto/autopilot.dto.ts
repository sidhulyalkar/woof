import { Transform } from 'class-transformer';
import {
  IsDateString,
  IsIn,
  IsInt,
  IsObject,
  IsOptional,
  IsString,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';
import { AUTOPILOT_OBSERVATION_KINDS, CARE_REMINDER_KINDS } from '../autopilot.types';

export class IngestTrackerObservationDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(160)
  externalEventId!: string;

  @Transform(({ value }) => String(value).toUpperCase())
  @IsIn(AUTOPILOT_OBSERVATION_KINDS)
  kind!: (typeof AUTOPILOT_OBSERVATION_KINDS)[number];

  @IsDateString()
  observedAt!: string;

  @IsObject()
  payload!: Record<string, unknown>;
}

export class CreateCareReminderDto {
  @IsOptional()
  @IsPetIdentifier()
  petId?: string;

  @Transform(({ value }) => String(value).toUpperCase())
  @IsIn(CARE_REMINDER_KINDS)
  kind!: (typeof CARE_REMINDER_KINDS)[number];

  @IsString()
  @MaxLength(120)
  title!: string;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  detail?: string;

  @IsDateString()
  dueAt!: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(365)
  repeatEveryDays?: number;
}
