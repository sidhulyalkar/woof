import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  IsBoolean,
  IsIn,
  IsObject,
  IsOptional,
  IsString,
  Matches,
  MaxLength,
  MinLength,
} from 'class-validator';
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
    example: 'south-bay-ca',
    description: 'Coarse, user-chosen locality key. Never an address, coordinate, or route trace.',
  })
  @IsString()
  @MinLength(2)
  @MaxLength(64)
  @Matches(/^[a-z0-9]+(?:-[a-z0-9]+)*$/)
  regionKey!: string;
}

export class HumanSkillChallengeParamDto {
  @ApiProperty({ enum: HUMAN_SKILL_CHALLENGES })
  @IsIn(HUMAN_SKILL_CHALLENGES)
  challengeKey!: (typeof HUMAN_SKILL_CHALLENGES)[number];
}
