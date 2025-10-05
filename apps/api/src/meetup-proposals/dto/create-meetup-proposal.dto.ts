import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsDateString, IsObject, IsOptional } from 'class-validator';

export class CreateMeetupProposalDto {
  @ApiProperty({ description: 'User ID of the recipient' })
  @IsString()
  recipientId: string;

  @ApiProperty({ description: 'Suggested time for the meetup' })
  @IsDateString()
  suggestedTime: string;

  @ApiProperty({
    description: 'Suggested venue (name, lat, lng, type, address)',
    example: {
      name: 'Riverside Dog Park',
      lat: 37.7749,
      lng: -122.4194,
      type: 'park',
      address: '123 Main St, San Francisco, CA',
    },
  })
  @IsObject()
  suggestedVenue: {
    name: string;
    lat: number;
    lng: number;
    type: string;
    address?: string;
  };

  @ApiProperty({ description: 'Optional notes for the proposal', required: false })
  @IsOptional()
  @IsString()
  notes?: string;
}
