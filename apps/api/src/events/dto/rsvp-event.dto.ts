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

  @ApiProperty({ description: 'Pet density', enum: ['too_crowded', 'just_right', 'too_few'], required: false })
  @IsOptional()
  @IsString()
  petDensity?: string;

  @ApiProperty({ description: 'Surface type', enum: ['grass', 'dirt', 'pavement', 'sand'], required: false })
  @IsOptional()
  @IsString()
  surfaceType?: string;

  @ApiProperty({ description: 'Crowding level', enum: ['not_crowded', 'moderate', 'crowded'], required: false })
  @IsOptional()
  @IsString()
  crowding?: string;

  @ApiProperty({ description: 'Noise level', enum: ['quiet', 'moderate', 'loud'], required: false })
  @IsOptional()
  @IsString()
  noiseLevel?: string;

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
