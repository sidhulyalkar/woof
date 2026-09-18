import { Type } from 'class-transformer';
import { IsUUID, Matches, ValidateNested } from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';
import { DailySignalsAnswersDto } from './daily-signals.dto';

const HOUSEHOLD_IDENTIFIER_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class DailySignalsCurrentQueryDto {
  @Matches(HOUSEHOLD_IDENTIFIER_PATTERN, {
    message: 'householdId must be a UUID-shaped identifier',
  })
  householdId!: string;

  @IsPetIdentifier()
  petId!: string;
}

export class CorrectDailySignalsDto extends DailySignalsCurrentQueryDto {
  @IsUUID()
  expectedCurrentCareEventId!: string;

  @ValidateNested()
  @Type(() => DailySignalsAnswersDto)
  signals!: DailySignalsAnswersDto;
}
