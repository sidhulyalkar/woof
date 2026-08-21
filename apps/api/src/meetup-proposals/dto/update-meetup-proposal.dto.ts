import {
  ArrayMaxSize,
  IsArray,
  IsBoolean,
  IsEnum,
  IsInt,
  IsOptional,
  IsString,
  Max,
  MaxLength,
  Min,
} from 'class-validator';

export enum MeetupProposalStatus {
  PENDING = 'pending',
  ACCEPTED = 'accepted',
  DECLINED = 'declined',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
}

export class UpdateMeetupProposalDto {
  @IsEnum(MeetupProposalStatus)
  status!: MeetupProposalStatus;
}

export class CompleteMeetupDto {
  @IsBoolean()
  occurred!: boolean;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(5)
  rating?: number;

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(8)
  @IsString({ each: true })
  @MaxLength(60, { each: true })
  feedbackTags?: string[];

  @IsOptional()
  @IsBoolean()
  checklistOk?: boolean;

  @IsOptional()
  @IsString()
  @MaxLength(1200)
  notes?: string;
}
