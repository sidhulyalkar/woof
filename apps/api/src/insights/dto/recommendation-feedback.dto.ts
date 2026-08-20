import { IsIn, IsOptional, IsString, MaxLength } from 'class-validator';

export class RecommendationFeedbackDto {
  @IsString()
  @MaxLength(120)
  recommendationId!: string;

  @IsString()
  @IsIn(['shown', 'accepted', 'dismissed', 'completed'])
  outcome!: 'shown' | 'accepted' | 'dismissed' | 'completed';

  @IsOptional()
  @IsString()
  @MaxLength(40)
  category?: string;
}
