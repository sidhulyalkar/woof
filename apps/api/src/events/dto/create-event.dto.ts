import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsOptional, IsDateString, IsNumber, IsArray } from 'class-validator';

export class CreateEventDto {
  @ApiProperty({ description: 'Event title' })
  @IsString()
  title: string;

  @ApiProperty({ description: 'Event description', required: false })
  @IsOptional()
  @IsString()
  description?: string;

  @ApiProperty({ description: 'Event type', example: 'group_walk' })
  @IsString()
  type: string;

  @ApiProperty({ description: 'Event start time' })
  @IsDateString()
  startTime: string;

  @ApiProperty({ description: 'Event end time', required: false })
  @IsOptional()
  @IsDateString()
  endTime?: string;

  @ApiProperty({ description: 'Location name', required: false })
  @IsOptional()
  @IsString()
  locationName?: string;

  @ApiProperty({ description: 'Latitude' })
  @IsNumber()
  lat: number;

  @ApiProperty({ description: 'Longitude' })
  @IsNumber()
  lng: number;

  @ApiProperty({ description: 'Maximum attendees', required: false })
  @IsOptional()
  @IsNumber()
  maxAttendees?: number;

  @ApiProperty({ description: 'Tags for the event', required: false, type: [String] })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  tags?: string[];
}
