import { IsBoolean, IsIn, IsOptional, IsString, MaxLength } from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';

export class QuestInteractionDto {
  @IsPetIdentifier()
  petId!: string;
}

export class CompleteQuestDto {
  @IsPetIdentifier()
  petId!: string;

  @IsIn(['loved_it', 'comfortable', 'not_their_thing'])
  dogExperience!: 'loved_it' | 'comfortable' | 'not_their_thing';

  @IsIn(['great', 'fine', 'a_lot_today'])
  ownerExperience!: 'great' | 'fine' | 'a_lot_today';

  @IsOptional()
  @IsBoolean()
  safeOptOut?: boolean;

  @IsOptional()
  @IsString()
  memoryAssetId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(280)
  note?: string;
}
