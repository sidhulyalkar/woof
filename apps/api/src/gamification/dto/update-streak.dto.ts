import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsDateString } from 'class-validator';

export class UpdateStreakDto {
  @ApiProperty({ description: 'User ID to update streak for' })
  @IsString()
  userId: string;

  @ApiProperty({ description: 'Activity date', example: '2025-01-15' })
  @IsDateString()
  activityDate: string;
}
