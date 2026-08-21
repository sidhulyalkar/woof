import { Type } from 'class-transformer';
import {
  IsDateString,
  IsOptional,
  IsString,
  IsUUID,
  MaxLength,
  ValidateNested,
} from 'class-validator';

export class SuggestedVenueDto {
  @IsString()
  @MaxLength(120)
  name!: string;

  @IsString()
  @MaxLength(60)
  type!: string;

  @IsOptional()
  @IsString()
  @MaxLength(120)
  area?: string;
}

export class CreateMeetupProposalDto {
  @IsUUID()
  recipientId!: string;

  @IsDateString()
  suggestedTime!: string;

  @ValidateNested()
  @Type(() => SuggestedVenueDto)
  suggestedVenue!: SuggestedVenueDto;

  @IsOptional()
  @IsString()
  @MaxLength(600)
  notes?: string;
}
