import { IsString, IsEnum, IsObject, ValidateNested, IsOptional } from 'class-validator';
import { Type } from 'class-transformer';
import { ApiProperty } from '@nestjs/swagger';

export enum NudgeType {
  MEETUP = 'meetup',
  SERVICE = 'service',
  EVENT = 'event',
  ACHIEVEMENT = 'achievement',
}

export enum NudgeReason {
  PROXIMITY = 'proximity',
  CHAT_ACTIVITY = 'chat_activity',
  MUTUAL_AVAILABILITY = 'mutual_availability',
  GOAL_ACHIEVEMENT = 'goal_achievement',
}

export class NudgeContext {
  @ApiProperty({ required: false })
  @IsOptional()
  @IsString()
  targetUserId?: string;

  @ApiProperty({ required: false })
  @IsOptional()
  @IsObject()
  location?: { lat: number; lng: number };

  @ApiProperty({ enum: NudgeReason })
  @IsEnum(NudgeReason)
  reason: NudgeReason;

  @ApiProperty({ required: false })
  @IsOptional()
  @IsString()
  message?: string;

  @ApiProperty({ required: false })
  @IsOptional()
  @IsObject()
  metadata?: Record<string, any>;
}

export class CreateNudgeDto {
  @ApiProperty()
  @IsString()
  userId: string;

  @ApiProperty({ enum: NudgeType })
  @IsEnum(NudgeType)
  type: NudgeType;

  @ApiProperty({ type: NudgeContext })
  @ValidateNested()
  @Type(() => NudgeContext)
  context: NudgeContext;
}
