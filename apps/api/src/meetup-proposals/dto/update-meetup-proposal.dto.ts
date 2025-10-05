import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsEnum, IsOptional, IsInt, Min, Max, IsArray, IsBoolean } from 'class-validator';

export enum MeetupProposalStatus {
  PENDING = 'pending',
  ACCEPTED = 'accepted',
  DECLINED = 'declined',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
}

export class UpdateMeetupProposalDto {
  @ApiProperty({ enum: MeetupProposalStatus, required: false })
  @IsOptional()
  @IsEnum(MeetupProposalStatus)
  status?: MeetupProposalStatus;
}

export class CompleteMeetupDto {
  @ApiProperty({ description: 'Did the meetup actually occur?', example: true })
  @IsBoolean()
  occurred: boolean;

  @ApiProperty({ description: 'Rating 1-5 stars', minimum: 1, maximum: 5, required: false })
  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(5)
  rating?: number;

  @ApiProperty({
    description: 'Feedback tags',
    example: ['energy_match', 'great_time', 'owner_friendly'],
    required: false,
  })
  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  feedbackTags?: string[];

  @ApiProperty({ description: 'Was the meet-safe checklist followed?', required: false })
  @IsOptional()
  @IsBoolean()
  checklistOk?: boolean;

  @ApiProperty({ description: 'Additional notes', required: false })
  @IsOptional()
  @IsString()
  notes?: string;
}
