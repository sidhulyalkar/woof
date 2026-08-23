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

export enum DogMeetupExperience {
  LOVED_IT = 'loved_it',
  COMFORTABLE = 'comfortable',
  NOT_THEIR_THING = 'not_their_thing',
}

export enum OwnerMeetupExperience {
  GREAT = 'great',
  FINE = 'fine',
  A_LOT_TODAY = 'a_lot_today',
}

export enum MeetAgainChoice {
  YES = 'yes',
  MAYBE = 'maybe',
  NO = 'no',
}

export class UpdateMeetupProposalDto {
  @IsEnum(MeetupProposalStatus)
  status!: MeetupProposalStatus;
}

export class CompleteMeetupDto {
  @IsBoolean()
  occurred!: boolean;

  @IsOptional()
  @IsEnum(DogMeetupExperience)
  dogExperience?: DogMeetupExperience;

  @IsOptional()
  @IsEnum(OwnerMeetupExperience)
  ownerExperience?: OwnerMeetupExperience;

  @IsOptional()
  @IsEnum(MeetAgainChoice)
  meetAgain?: MeetAgainChoice;

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
