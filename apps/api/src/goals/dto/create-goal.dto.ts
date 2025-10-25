import { IsString, IsNumber, IsOptional, IsBoolean, IsDateString, IsEnum } from 'class-validator';

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
  @IsString()
  petId: string;

  @IsEnum(GoalType)
  goalType: GoalType;

  @IsEnum(GoalPeriod)
  period: GoalPeriod;

  @IsNumber()
  targetNumber: number;

  @IsString()
  targetUnit: string; // km, minutes, steps, count, kcal, friends

  @IsDateString()
  startDate: string;

  @IsDateString()
  endDate: string;

  @IsOptional()
  @IsString()
  reminderTime?: string; // HH:MM format

  @IsOptional()
  @IsBoolean()
  isRecurring?: boolean;

  @IsOptional()
  metadata?: any; // Additional goal-specific data
}
