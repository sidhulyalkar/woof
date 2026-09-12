import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsBoolean,
  IsIn,
  IsObject,
  IsOptional,
  IsString,
  MaxLength,
  MinLength,
} from 'class-validator';
import { PACK_COARSE_REGION_IDS, type PackCoarseRegionId } from '../pack-locality.catalog';
import { HUMAN_SKILL_CHALLENGES } from '../social-adventure.policy';

export class UpdateSocialAdventurePreferencesDto {
  @ApiProperty({ description: 'Opt in to appearing on the global Social Adventure leaderboard.' })
  @IsBoolean()
  globalLeaderboardOptIn!: boolean;
}

export class CreateSocialShareDto {
  @ApiProperty({ enum: ['CARE_EVENT', 'HUMAN_SKILL_ATTEMPT'] })
  @IsIn(['CARE_EVENT', 'HUMAN_SKILL_ATTEMPT'])
  sourceType!: 'CARE_EVENT' | 'HUMAN_SKILL_ATTEMPT';

  @ApiProperty()
  @IsString()
  @MinLength(1)
  @MaxLength(128)
  sourceId!: string;

  @ApiPropertyOptional({ maxLength: 280 })
  @IsOptional()
  @IsString()
  @MaxLength(280)
  caption?: string;

  @ApiPropertyOptional({ enum: ['PUBLIC', 'PRIVATE'], default: 'PRIVATE' })
  @IsOptional()
  @IsIn(['PUBLIC', 'PRIVATE'])
  visibility?: 'PUBLIC' | 'PRIVATE';
}

export class SocialReactionDto {
  @ApiProperty({
    enum: ['NICE_READ', 'GOOD_CALL', 'TRYING_THIS', 'ADVENTURE_INSPIRATION', 'CHEER'],
  })
  @IsIn(['NICE_READ', 'GOOD_CALL', 'TRYING_THIS', 'ADVENTURE_INSPIRATION', 'CHEER'])
  reaction!: 'NICE_READ' | 'GOOD_CALL' | 'TRYING_THIS' | 'ADVENTURE_INSPIRATION' | 'CHEER';
}

export class CompleteHumanSkillAttemptDto {
  @ApiProperty({
    description:
      'Challenge response only. Scores, correct answers, target timing and leaderboard value are server-authored.',
  })
  @IsObject()
  response!: Record<string, unknown>;
}

export class CreatePackDto {
  @ApiProperty({ example: 'South Bay Adventure Pack' })
  @IsString()
  @MinLength(2)
  @MaxLength(64)
  name!: string;

  @ApiProperty({
    enum: PACK_COARSE_REGION_IDS,
    example: 'us-ca-south-bay',
    description:
      'Server-approved broad locality identity. Clients select this value from the coarse-region catalog; arbitrary addresses, venues, coordinates, routes, and free-form locality text are rejected.',
  })
  @IsIn(PACK_COARSE_REGION_IDS)
  regionKey!: PackCoarseRegionId;
}

export class UpdatePackLocalityDto {
  @ApiProperty({
    enum: PACK_COARSE_REGION_IDS,
    example: 'us-ca-south-bay',
    description:
      'One approved broad locality used to repair a legacy Pack whose former free-form locality was discarded.',
  })
  @IsIn(PACK_COARSE_REGION_IDS)
  regionKey!: PackCoarseRegionId;
}

export class HumanSkillChallengeParamDto {
  @ApiProperty({ enum: HUMAN_SKILL_CHALLENGES })
  @IsIn(HUMAN_SKILL_CHALLENGES)
  challengeKey!: (typeof HUMAN_SKILL_CHALLENGES)[number];
}
