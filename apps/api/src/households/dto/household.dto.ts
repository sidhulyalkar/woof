import { IsOptional, IsString, Length } from 'class-validator';

export class UpdateHouseholdDto {
  @IsOptional()
  @IsString()
  @Length(1, 80)
  name?: string;

  @IsOptional()
  @IsString()
  @Length(1, 80)
  timezone?: string;
}
