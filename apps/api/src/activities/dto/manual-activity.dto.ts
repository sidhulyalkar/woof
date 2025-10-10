import { IsString, IsNumber, IsOptional, IsDateString, IsEnum } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export enum ActivityType {
  WALK = 'walk',
  RUN = 'run',
  PLAY = 'play',
  TRAINING = 'training',
  GROOMING = 'grooming',
  VET_VISIT = 'vet_visit',
  OTHER = 'other',
}

export class ManualActivityDto {
  @ApiProperty({ description: 'Pet ID' })
  @IsString()
  petId: string;

  @ApiProperty({ enum: ActivityType, description: 'Type of activity' })
  @IsEnum(ActivityType)
  type: ActivityType;

  @ApiProperty({ description: 'Activity date and time' })
  @IsDateString()
  datetime: string;

  @ApiProperty({ description: 'Duration in minutes', required: false })
  @IsOptional()
  @IsNumber()
  duration?: number;

  @ApiProperty({ description: 'Distance in kilometers', required: false })
  @IsOptional()
  @IsNumber()
  distance?: number;

  @ApiProperty({ description: 'Calories burned', required: false })
  @IsOptional()
  @IsNumber()
  calories?: number;

  @ApiProperty({ description: 'Activity notes', required: false })
  @IsOptional()
  @IsString()
  notes?: string;

  @ApiProperty({ description: 'Location name', required: false })
  @IsOptional()
  @IsString()
  location?: string;

  @ApiProperty({ description: 'Photo URLs', required: false })
  @IsOptional()
  photos?: string[];
}
