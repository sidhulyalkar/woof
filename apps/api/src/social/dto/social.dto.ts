import { ApiProperty, ApiPropertyOptional, PartialType } from '@nestjs/swagger';
import {
  ArrayMaxSize,
  IsArray,
  IsIn,
  IsOptional,
  IsString,
  IsUrl,
  IsUUID,
  MaxLength,
  MinLength,
} from 'class-validator';

export class CreatePostDto {
  @ApiPropertyOptional({ example: 'Golden-hour trail walk with Shasta.' })
  @IsOptional()
  @IsString()
  @MaxLength(2000)
  text?: string;

  @ApiPropertyOptional({ type: [String], maxItems: 8 })
  @IsOptional()
  @IsArray()
  @ArrayMaxSize(8)
  @IsUrl({ require_tld: false }, { each: true })
  mediaUrls?: string[];

  @ApiPropertyOptional()
  @IsOptional()
  @IsUUID()
  petId?: string;

  @ApiPropertyOptional()
  @IsOptional()
  @IsUUID()
  activityId?: string;

  @ApiPropertyOptional({ enum: ['PUBLIC', 'FRIENDS_ONLY', 'PRIVATE'] })
  @IsOptional()
  @IsIn(['PUBLIC', 'FRIENDS_ONLY', 'PRIVATE'])
  visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE';
}

export class UpdatePostDto extends PartialType(CreatePostDto) {}

export class CreateCommentDto {
  @ApiProperty({ example: 'This trail looks perfect for a weekend meetup.' })
  @IsString()
  @MinLength(1)
  @MaxLength(1000)
  text!: string;
}

export class UpdateCommentDto {
  @ApiProperty({ example: 'Updated comment text.' })
  @IsString()
  @MinLength(1)
  @MaxLength(1000)
  text!: string;
}
