import { ApiPropertyOptional } from '@nestjs/swagger';
import { IsOptional, IsUUID } from 'class-validator';

export class ConciergeQueryDto {
  @ApiPropertyOptional({ description: 'Optional owned pet to use for today context' })
  @IsOptional()
  @IsUUID()
  petId?: string;
}
