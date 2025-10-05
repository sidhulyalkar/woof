import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsInt, Min, IsOptional } from 'class-validator';

export class AwardPointsDto {
  @ApiProperty({ description: 'User ID to award points to' })
  @IsString()
  userId: string;

  @ApiProperty({ description: 'Number of points to award', minimum: 1 })
  @IsInt()
  @Min(1)
  points: number;

  @ApiProperty({ description: 'Reason for awarding points', example: 'first_match' })
  @IsString()
  reason: string;

  @ApiProperty({ description: 'Related entity ID (e.g., meetupId, eventId)', required: false })
  @IsOptional()
  @IsString()
  relatedEntityId?: string;
}
