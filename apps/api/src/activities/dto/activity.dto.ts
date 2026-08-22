import { Transform } from 'class-transformer';
import {
  ArrayMaxSize,
  ArrayUnique,
  IsArray,
  IsDateString,
  IsIn,
  IsObject,
  IsOptional,
  IsString,
  IsUUID,
} from 'class-validator';

const ACTIVITY_TYPES = [
  'WALK',
  'RUN',
  'PLAY',
  'HIKE',
  'TRAINING',
  'GROOMING',
  'VET_VISIT',
  'ENRICHMENT',
  'SCENT',
  'PUZZLE',
  'SOCIAL',
  'MEETUP',
  'PARALLEL_WALK',
  'RECOVERY',
  'REST',
  'DECOMPRESSION',
  'OTHER',
];

export class CreateActivityDto {
  /** Legacy single-pet field. dogOS clients should prefer petIds. */
  @IsOptional()
  @IsUUID()
  petId?: string;

  @IsOptional()
  @IsArray()
  @ArrayUnique()
  @ArrayMaxSize(16)
  @IsUUID('4', { each: true })
  petIds?: string[];

  @IsOptional()
  @IsUUID()
  householdId?: string;

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
  @IsArray()
  @ArrayUnique()
  @ArrayMaxSize(16)
  @IsUUID('4', { each: true })
  petIds?: string[];

  @IsOptional()
  @IsUUID()
  householdId?: string;

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
