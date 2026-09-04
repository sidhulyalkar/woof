import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional } from 'class-validator';
import { IsPetIdentifier } from '../../common/validation/pet-identifier';

export class ConciergeQueryDto {
  @ApiPropertyOptional({ description: 'Optional owned pet to use for today context' })
  @IsOptional()
  @IsPetIdentifier()
  petId?: string;
}
