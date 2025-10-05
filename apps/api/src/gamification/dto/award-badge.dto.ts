import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsEnum } from 'class-validator';

export enum BadgeType {
  FIRST_MATCH = 'first_match',
  FIRST_MEETUP = 'first_meetup',
  SUPER_SOCIAL = 'super_social', // 10+ meetups
  EVENT_HOST = 'event_host',
  VERIFIED_OWNER = 'verified_owner',
  EARLY_ADOPTER = 'early_adopter',
  STREAK_MASTER = 'streak_master', // 4+ week streak
  COMMUNITY_BUILDER = 'community_builder',
}

export class AwardBadgeDto {
  @ApiProperty({ description: 'User ID to award badge to' })
  @IsString()
  userId: string;

  @ApiProperty({ enum: BadgeType, description: 'Type of badge to award' })
  @IsEnum(BadgeType)
  badgeType: BadgeType;
}
