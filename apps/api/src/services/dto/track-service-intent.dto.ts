import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsEnum, IsOptional, IsBoolean } from 'class-validator';

export enum ServiceAction {
  VIEW = 'view',
  TAP_CALL = 'tap_call',
  TAP_DIRECTIONS = 'tap_directions',
  TAP_WEBSITE = 'tap_website',
  TAP_BOOK = 'tap_book',
}

export class TrackServiceIntentDto {
  @ApiProperty({ description: 'Business ID' })
  @IsString()
  businessId: string;

  @ApiProperty({ enum: ServiceAction, description: 'Action taken by user' })
  @IsEnum(ServiceAction)
  action: ServiceAction;
}

export class ServiceIntentFollowupDto {
  @ApiProperty({ description: 'Did the user book the service?', example: true })
  @IsBoolean()
  converted: boolean;

  @ApiProperty({ description: 'Optional notes from the user', required: false })
  @IsOptional()
  @IsString()
  notes?: string;
}
