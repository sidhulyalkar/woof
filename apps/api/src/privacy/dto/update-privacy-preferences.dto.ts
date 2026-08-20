import { IsBoolean, IsIn, IsInt, IsOptional, Max, Min } from 'class-validator';
import { LOCATION_SHARING_MODES, MeetupLocationSharing } from '../privacy.types';

export class UpdatePrivacyPreferencesDto {
  @IsOptional()
  @IsBoolean()
  preciseLocation?: boolean;

  @IsOptional()
  @IsBoolean()
  proximitySuggestions?: boolean;

  @IsOptional()
  @IsBoolean()
  shareActivityRoutes?: boolean;

  @IsOptional()
  @IsIn(LOCATION_SHARING_MODES)
  meetupLocationSharing?: MeetupLocationSharing;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(24)
  locationRetentionHours?: number;
}
