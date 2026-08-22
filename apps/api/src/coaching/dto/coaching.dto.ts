import { Transform } from 'class-transformer';
import {
  ArrayMaxSize,
  IsArray,
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

export const TRAINING_REWARD_TYPES = [
  'food',
  'play',
  'praise',
  'access',
  'environmental',
  'other',
] as const;

export const TRAINING_STRESS_SIGNALS = [
  'look-away',
  'lip-lick',
  'yawning',
  'panting',
  'freezing',
  'cowering',
  'escape-attempt',
  'growling',
  'hiding',
  'tail-tucked',
] as const;

export class CreateTrainingPlanDto {
  @IsUUID()
  petId!: string;

  @IsString()
  @MaxLength(64)
  templateId!: string;
}

export class RecordTrainingSessionDto {
  @IsInt()
  @Min(1)
  @Max(20)
  attempts!: number;

  @IsInt()
  @Min(0)
  @Max(20)
  successes!: number;

  @IsInt()
  @Min(20)
  @Max(900)
  durationSeconds!: number;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(5)
  distractionLevel?: number;

  @IsString()
  @IsIn(TRAINING_REWARD_TYPES)
  rewardType!: (typeof TRAINING_REWARD_TYPES)[number];

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(6)
  @IsIn(TRAINING_STRESS_SIGNALS, { each: true })
  stressSignals?: string[];

  @IsOptional()
  @IsBoolean()
  stoppedEarly?: boolean;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  notes?: string;
}

export class UpdateTrainingPlanStatusDto {
  @Transform(({ value }) => String(value).toUpperCase())
  @IsString()
  @IsIn(['ACTIVE', 'PAUSED'])
  status!: 'ACTIVE' | 'PAUSED';
}
