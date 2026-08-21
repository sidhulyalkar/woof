import { ApiProperty } from '@nestjs/swagger';
import {
  IsDateString,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  Min,
} from 'class-validator';

export class TrackLocationDto {
  @ApiProperty({ description: 'Latitude', example: 37.7749 })
  @IsNumber()
  @Min(-90)
  @Max(90)
  lat!: number;

  @ApiProperty({ description: 'Longitude', example: -122.4194 })
  @IsNumber()
  @Min(-180)
  @Max(180)
  lng!: number;

  @ApiProperty({
    description: 'Client observation timestamp. Server rejects stale/future pings.',
    required: false,
  })
  @IsOptional()
  @IsDateString()
  timestamp?: string;

  @ApiProperty({ description: 'Activity type', required: false, example: 'walk' })
  @IsOptional()
  @IsString()
  activityType?: string;
}
