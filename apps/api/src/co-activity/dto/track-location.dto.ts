import { ApiProperty } from '@nestjs/swagger';
import { IsNumber, IsString, IsDateString, IsOptional } from 'class-validator';

export class TrackLocationDto {
  @ApiProperty({ description: 'Latitude', example: 37.7749 })
  @IsNumber()
  lat: number;

  @ApiProperty({ description: 'Longitude', example: -122.4194 })
  @IsNumber()
  lng: number;

  @ApiProperty({ description: 'Timestamp of location ping', example: '2025-01-15T10:30:00Z' })
  @IsDateString()
  timestamp: string;

  @ApiProperty({ description: 'Activity type', required: false, example: 'walk' })
  @IsOptional()
  @IsString()
  activityType?: string;
}
