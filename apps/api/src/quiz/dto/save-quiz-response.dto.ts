import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsObject, IsOptional, IsString, IsUUID, MinLength } from 'class-validator';

export class SaveQuizResponseDto {
  @ApiProperty({ example: 'onboarding-1724131200000' })
  @IsString()
  @MinLength(8)
  sessionId!: string;

  @ApiPropertyOptional({ description: 'Pet this preference session primarily describes.' })
  @IsOptional()
  @IsUUID()
  petId?: string;

  @ApiProperty({
    description:
      'Question IDs mapped to a single answer, numeric scale answer, or an array of selections.',
    example: {
      activity_level: 4,
      schedule: ['Weekday mornings', 'Weekends'],
    },
  })
  @IsObject()
  responses!: Record<string, string | string[] | number>;
}
