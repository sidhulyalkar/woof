import { IsBoolean, IsIn, IsOptional, IsString, MaxLength } from 'class-validator';

export class QuestInteractionDto {
  @IsString()
  petId!: string;
}

export class CompleteQuestDto {
  @IsString()
  petId!: string;

  @IsIn(['loved_it', 'comfortable', 'not_their_thing'])
  dogExperience!: 'loved_it' | 'comfortable' | 'not_their_thing';

  @IsIn(['great', 'fine', 'a_lot_today'])
  ownerExperience!: 'great' | 'fine' | 'a_lot_today';

  @IsOptional()
  @IsBoolean()
  safeOptOut?: boolean;

  @IsOptional()
  @IsBoolean()
  newPlace?: boolean;

  @IsOptional()
  @IsString()
  memoryAssetId?: string;

  @IsOptional()
  @IsString()
  @MaxLength(280)
  note?: string;
}
