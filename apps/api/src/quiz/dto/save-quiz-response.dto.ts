import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsObject, IsOptional, IsString, MinLength } from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';

export class SaveQuizResponseDto {
  @ApiProperty({ example: 'onboarding-1724131200000' })
  @IsString()
  @MinLength(8)
  sessionId!: string;

  @ApiPropertyOptional({ description: 'Pet this preference session primarily describes.' })
  @IsOptional()
  @IsPetIdentifier()
  petId?: string;

  @ApiProperty({
    description: 'Question IDs mapped to a single answer or an array of selected answers.',
    example: {
      activity_level: 'High - Very active lifestyle',
      schedule: ['Weekday mornings', 'Weekends'],
    },
  })
  @IsObject()
  responses!: Record<string, string | string[]>;
}
