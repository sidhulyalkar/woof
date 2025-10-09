import { ApiProperty } from '@nestjs/swagger';
import { IsEnum, IsOptional, IsInt, Min, Max, IsArray, IsString } from 'class-validator';

export enum RSVPStatus {
  GOING = 'going',
  MAYBE = 'maybe',
  NOT_GOING = 'not_going',
}

export class CreateRSVPDto {
  @ApiProperty({ enum: RSVPStatus, description: 'RSVP status' })
  @IsEnum(RSVPStatus)
  status: RSVPStatus;
}

export class UpdateRSVPDto extends CreateRSVPDto {}

export class EventFeedbackDto {
  @ApiProperty({ description: 'Vibe score 1-5', minimum: 1, maximum: 5 })
  @IsInt()
  @Min(1)
  @Max(5)
  vibeScore: number;

  @ApiProperty({ description: 'Pet density 1-5', minimum: 1, maximum: 5 })
  @IsInt()
  @Min(1)
  @Max(5)
  petDensity: number;

  @ApiProperty({ description: 'Venue quality 1-5', minimum: 1, maximum: 5 })
  @IsInt()
  @Min(1)
  @Max(5)
  venueQuality: number;

  @ApiProperty({ description: 'Feedback tags', type: [String], required: false })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  tags?: string[];

  @ApiProperty({ description: 'Optional notes', required: false })
  @IsOptional()
  @IsString()
  notes?: string;
}
