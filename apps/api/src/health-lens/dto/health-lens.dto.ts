import { Transform, Type } from 'class-transformer';
import {
  IsBoolean,
  IsIn,
  IsInt,
  IsOptional,
  IsString,
  IsUUID,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';

export const HEALTH_BODY_AREAS = [
  'general',
  'skin',
  'eye',
  'ear',
  'mouth-teeth',
  'paw-limb',
  'abdomen',
  'stool-urine',
  'movement-gait',
  'wound',
  'other',
] as const;

export const HEALTH_CHANGE_LEVELS = ['normal', 'mild-change', 'major-change', 'unknown'] as const;

export class AnalyzePetHealthDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(1200)
  concern!: string;

  @IsOptional()
  @IsString()
  @IsIn(HEALTH_BODY_AREAS)
  bodyArea?: (typeof HEALTH_BODY_AREAS)[number];

  @IsOptional()
  @IsString()
  @MaxLength(80)
  onset?: string;

  @IsOptional()
  @IsString()
  @IsIn(HEALTH_CHANGE_LEVELS)
  appetite?: (typeof HEALTH_CHANGE_LEVELS)[number];

  @IsOptional()
  @IsString()
  @IsIn(HEALTH_CHANGE_LEVELS)
  energy?: (typeof HEALTH_CHANGE_LEVELS)[number];

  @IsOptional()
  @IsString()
  @IsIn(HEALTH_CHANGE_LEVELS)
  breathing?: (typeof HEALTH_CHANGE_LEVELS)[number];

  @IsOptional()
  @IsString()
  @IsIn(HEALTH_CHANGE_LEVELS)
  bathroom?: (typeof HEALTH_CHANGE_LEVELS)[number];

  @IsOptional()
  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  saveToTimeline?: boolean;
}

export class FollowUpHealthDto {
  @IsUUID()
  assessmentId!: string;

  @IsString()
  @MaxLength(1200)
  message!: string;
}

export class HealthTimelineQueryDto {
  @IsPetIdentifier()
  petId!: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(50)
  limit?: number;
}
