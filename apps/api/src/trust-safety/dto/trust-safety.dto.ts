import {
  ArrayMaxSize,
  IsArray,
  IsIn,
  IsOptional,
  IsString,
  IsUrl,
  IsUUID,
  MaxLength,
} from 'class-validator';

export const REPORT_REASONS = [
  'inappropriate_behavior',
  'safety_concern',
  'spam',
  'fake_profile',
  'harassment',
] as const;

export class BlockUserDto {
  @IsUUID()
  blockedUserId!: string;

  @IsOptional()
  @IsString()
  @MaxLength(240)
  reason?: string;
}

export class ReportUserDto {
  @IsUUID()
  reportedUserId!: string;

  @IsIn(REPORT_REASONS)
  reason!: (typeof REPORT_REASONS)[number];

  @IsOptional()
  @IsString()
  @MaxLength(2000)
  description?: string;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(5)
  @IsUrl({}, { each: true })
  evidence?: string[];
}
