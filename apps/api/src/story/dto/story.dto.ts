import {
  IsDateString,
  IsIn,
  IsInt,
  IsOptional,
  IsString,
  IsUUID,
  Max,
  MaxLength,
  Min,
} from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';
import { STORY_SOURCE_TYPES } from '../story.types';

export class StoryQueryDto {
  @IsOptional()
  @IsPetIdentifier()
  petId?: string;

  @IsOptional()
  @IsDateString()
  before?: string;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(100)
  limit?: number;
}

export class UpdateStoryCurationDto {
  @IsIn(STORY_SOURCE_TYPES)
  sourceType!: 'ACTIVITY' | 'CARE_EVENT' | 'MEDIA';

  @IsUUID()
  sourceId!: string;

  @IsIn(['SAVE', 'CLEAR'])
  action!: 'SAVE' | 'CLEAR';

  @IsOptional()
  @IsString()
  @MaxLength(500)
  note?: string;
}
