import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsIn, IsOptional, IsString, IsUrl, MaxLength, MinLength } from 'class-validator';

export class UpdateProfileDto {
  @ApiPropertyOptional({ example: 'trailpaws' })
  @IsOptional()
  @IsString()
  @MinLength(3)
  @MaxLength(30)
  handle?: string;

  @ApiPropertyOptional({ example: 'Husky owner who is usually up for trail days.' })
  @IsOptional()
  @IsString()
  @MaxLength(500)
  bio?: string;

  @ApiPropertyOptional({ example: 'https://cdn.example.com/users/avatar.jpg' })
  @IsOptional()
  @IsUrl({ require_tld: false })
  avatarUrl?: string;

  @ApiPropertyOptional({ enum: ['PUBLIC', 'FRIENDS_ONLY', 'PRIVATE'] })
  @IsOptional()
  @IsIn(['PUBLIC', 'FRIENDS_ONLY', 'PRIVATE'])
  visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE';
}
