import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import {
  ArrayMinSize,
  ArrayUnique,
  IsArray,
  IsISO8601,
  IsIn,
  IsOptional,
  IsString,
  Length,
  MaxLength,
} from 'class-validator';
import {
  CAREGIVER_CAPABILITIES,
  CAREGIVER_OBSERVATION_KINDS,
  type CaregiverCapability,
  type CaregiverObservationKind,
} from '../caregiver.policy';

export class IssueCaregiverGrantDto {
  @ApiProperty()
  @IsString()
  @Length(1, 128)
  petId!: string;

  @ApiProperty()
  @IsString()
  @Length(1, 128)
  recipientUserId!: string;

  @ApiProperty({ enum: CAREGIVER_CAPABILITIES, isArray: true })
  @IsArray()
  @ArrayMinSize(1)
  @ArrayUnique()
  @IsIn(CAREGIVER_CAPABILITIES, { each: true })
  capabilities!: CaregiverCapability[];

  @ApiProperty({ description: 'Bounded v1 expiry as an ISO-8601 instant.' })
  @IsISO8601({ strict: true })
  expiresAt!: string;

  @ApiProperty({ description: 'Caller-stable replay identity for grant issuance.' })
  @IsString()
  @Length(8, 128)
  requestKey!: string;
}

export class CreateCaregiverObservationDto {
  @ApiProperty({ enum: CAREGIVER_OBSERVATION_KINDS })
  @IsIn(CAREGIVER_OBSERVATION_KINDS)
  kind!: CaregiverObservationKind;

  @ApiProperty()
  @IsString()
  @Length(1, 240)
  summary!: string;

  @ApiPropertyOptional()
  @IsOptional()
  @IsString()
  @MaxLength(500)
  note?: string;

  @ApiPropertyOptional({ description: 'Observed instant. Defaults to capture time.' })
  @IsOptional()
  @IsISO8601({ strict: true })
  observedAt?: string;
}
