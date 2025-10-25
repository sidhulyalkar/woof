import { IsString, IsNumber, IsOptional, IsEnum } from 'class-validator';

export enum GoalStatus {
  ACTIVE = 'ACTIVE',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  PAUSED = 'PAUSED',
}

export class UpdateGoalDto {
  @IsOptional()
  @IsNumber()
  targetNumber?: number;

  @IsOptional()
  @IsString()
  targetUnit?: string;

  @IsOptional()
  @IsEnum(GoalStatus)
  status?: GoalStatus;

  @IsOptional()
  @IsString()
  reminderTime?: string;

  @IsOptional()
  metadata?: any;
}
