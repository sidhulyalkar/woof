import {
  IsBoolean,
  IsDateString,
  IsEnum,
  IsNumber,
  IsObject,
  IsOptional,
  IsString,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';

export enum GoalType {
  DISTANCE = 'DISTANCE',
  TIME = 'TIME',
  STEPS = 'STEPS',
  ACTIVITIES = 'ACTIVITIES',
  CALORIES = 'CALORIES',
  SOCIAL = 'SOCIAL',
}

export enum GoalPeriod {
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY',
  CUSTOM = 'CUSTOM',
}

export class CreateGoalDto {
  @IsPetIdentifier()
  petId: string;

  @IsEnum(GoalType)
  goalType: GoalType;

  @IsEnum(GoalPeriod)
  period: GoalPeriod;

  @IsNumber()
  targetNumber: number;

  @IsString()
  targetUnit: string;

  @IsDateString()
  startDate: string;

  @IsDateString()
  endDate: string;

  @IsOptional()
  @IsString()
  reminderTime?: string;

  @IsOptional()
  @IsBoolean()
  isRecurring?: boolean;

  @IsOptional()
  @IsObject()
  metadata?: Record<string, unknown>;
}
