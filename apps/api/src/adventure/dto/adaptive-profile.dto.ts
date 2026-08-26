import { ArrayMaxSize, IsArray, IsIn, IsOptional, IsString, MaxLength } from 'class-validator';
import { PROFILE_DIMENSIONS } from '../profile-question-policy-v1.types';

const PROFILE_STATES = ['KNOWN', 'UNKNOWN'] as const;
const QUESTION_OUTCOMES = ['ANSWERED', 'NOT_SURE', 'SKIPPED'] as const;

export class RecordProfileQuestionResponseDto {
  @IsString()
  @MaxLength(128)
  responseId!: string;

  @IsString()
  @MaxLength(128)
  questionId!: string;

  @IsIn(QUESTION_OUTCOMES)
  outcome!: (typeof QUESTION_OUTCOMES)[number];

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(8)
  @IsString({ each: true })
  @MaxLength(80, { each: true })
  answers?: string[];
}

export class CorrectAdaptiveProfileDto {
  @IsString()
  @MaxLength(128)
  mutationId!: string;

  @IsIn(PROFILE_DIMENSIONS)
  dimension!: (typeof PROFILE_DIMENSIONS)[number];

  @IsIn(PROFILE_STATES)
  state!: (typeof PROFILE_STATES)[number];

  @IsOptional()
  @IsArray()
  @ArrayMaxSize(8)
  @IsString({ each: true })
  @MaxLength(80, { each: true })
  values?: string[];
}
