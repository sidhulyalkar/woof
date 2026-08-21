import { Transform, Type } from 'class-transformer';
import {
  IsBoolean,
  IsIn,
  IsInt,
  IsNumber,
  IsOptional,
  IsString,
  IsUUID,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import {
  BEHAVIOR_CONTEXTS,
  BEHAVIOR_PHASES,
  HANDLER_ACTIONS,
  type BehaviorContext,
  type BehaviorPhase,
  type HandlerAction,
} from '../behavior-vision.types';

const LEASH_STATES = ['off-leash', 'loose', 'tight', 'unknown'] as const;

export class AnalyzeBehaviorMediaDto {
  @IsUUID()
  petId!: string;

  @IsIn(BEHAVIOR_CONTEXTS)
  context!: BehaviorContext;

  @IsOptional()
  @IsString()
  @MaxLength(120)
  sessionKey?: string;

  @IsOptional()
  @IsIn(BEHAVIOR_PHASES)
  phase?: BehaviorPhase;

  @IsOptional()
  @IsIn(HANDLER_ACTIONS)
  handlerAction?: HandlerAction;

  @IsOptional()
  @IsIn(LEASH_STATES)
  leashState?: (typeof LEASH_STATES)[number];

  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  otherDogsPresent!: boolean;

  @IsOptional()
  @Type(() => Number)
  @IsNumber()
  @Min(0)
  @Max(500)
  otherDogDistanceMeters?: number;

  @IsOptional()
  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  familiarDog?: boolean;

  @IsOptional()
  @IsString()
  @MaxLength(800)
  ownerNote?: string;

  @IsOptional()
  @IsString()
  @MaxLength(500)
  question?: string;

  @IsOptional()
  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  saveToTimeline?: boolean;
}

export class BehaviorObservationFeedbackDto {
  @IsUUID()
  observationId!: string;

  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  accurate!: boolean;

  @IsOptional()
  @IsString()
  @MaxLength(800)
  note?: string;
}

export class BehaviorTimelineQueryDto {
  @IsUUID()
  petId!: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(100)
  limit?: number;
}
