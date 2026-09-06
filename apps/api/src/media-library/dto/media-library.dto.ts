import { Transform, Type } from 'class-transformer';
import {
  ArrayMaxSize,
  IsArray,
  IsBoolean,
  IsIn,
  IsInt,
  IsISO8601,
  IsOptional,
  IsString,
  IsUUID,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';
import { MEDIA_SOURCES, type MediaSource } from '../media-library.types';

export class CreateMediaUploadIntentDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(240)
  filename!: string;

  @IsString()
  @MaxLength(120)
  mimeType!: string;

  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(600 * 1024 * 1024)
  sizeBytes!: number;

  @IsOptional()
  @IsISO8601()
  capturedAt?: string;

  @IsOptional()
  @IsIn(MEDIA_SOURCES)
  source?: MediaSource;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(12)
  @IsString({ each: true })
  albumIds?: string[];

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(16)
  @IsString({ each: true })
  tags?: string[];

  @IsOptional()
  @IsString()
  @MaxLength(80)
  linkedObservationId?: string;
}

export class CompleteMediaUploadDto {
  @IsUUID()
  assetId!: string;

  @IsOptional()
  @IsString()
  @MaxLength(64)
  sha256?: string;
}

export class MediaLibraryQueryDto {
  @IsPetIdentifier()
  petId!: string;

  @IsOptional()
  @IsString()
  @MaxLength(80)
  albumId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(80)
  tag?: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(100)
  limit?: number;
}

export class CreateMediaAlbumDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(80)
  name!: string;

  @IsOptional()
  @IsString()
  @MaxLength(280)
  description?: string;
}

export class UpdateMediaAssetDto {
  @IsOptional()
  @Transform(({ value }) => value === true || value === 'true')
  @IsBoolean()
  favorite?: boolean;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(12)
  @IsString({ each: true })
  albumIds?: string[];

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(24)
  @IsString({ each: true })
  tags?: string[];
}

export class GooglePhotosPickerStartDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(4096)
  accessToken!: string;

  @IsOptional()
  @Type(() => Number)
  @IsInt()
  @Min(1)
  @Max(100)
  maxItemCount?: number;
}

export class GooglePhotosPickerImportDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(4096)
  accessToken!: string;

  @IsString()
  @MaxLength(512)
  sessionId!: string;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(12)
  @IsString({ each: true })
  albumIds?: string[];
}

export class GooglePhotosExportDto {
  @IsPetIdentifier()
  petId!: string;

  @IsString()
  @MaxLength(4096)
  accessToken!: string;

  @IsArray()
  @ArrayMaxSize(25)
  @IsUUID('4', { each: true })
  assetIds!: string[];
}

export class MediaExportManifestDto {
  @IsPetIdentifier()
  petId!: string;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(100)
  @IsUUID('4', { each: true })
  assetIds?: string[];
}
