import { Transform } from 'class-transformer';
import {
  IsDateString,
  IsIn,
  IsObject,
  IsOptional,
  IsString,
  IsUUID,
} from 'class-validator';

const ACTIVITY_TYPES = ['WALK', 'RUN', 'PLAY', 'HIKE', 'TRAINING', 'OTHER'];

export class CreateActivityDto {
  @IsOptional()
  @IsUUID()
  petId?: string;

  @IsOptional()
  @IsDateString()
  startedAt?: string;

  @IsOptional()
  @IsDateString()
  endedAt?: string;

  @Transform(({ value }) => String(value).toUpperCase())
  @IsString()
  @IsIn(ACTIVITY_TYPES)
  type!: string;

  @IsOptional()
  @IsObject()
  route?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  humanMetrics?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  petMetrics?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  jointMetrics?: Record<string, unknown>;
}

export class UpdateActivityDto {
  @IsOptional()
  @IsUUID()
  petId?: string;

  @IsOptional()
  @IsDateString()
  startedAt?: string;

  @IsOptional()
  @IsDateString()
  endedAt?: string;

  @IsOptional()
  @Transform(({ value }) => String(value).toUpperCase())
  @IsString()
  @IsIn(ACTIVITY_TYPES)
  type?: string;

  @IsOptional()
  @IsObject()
  route?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  humanMetrics?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  petMetrics?: Record<string, unknown>;

  @IsOptional()
  @IsObject()
  jointMetrics?: Record<string, unknown>;
}
