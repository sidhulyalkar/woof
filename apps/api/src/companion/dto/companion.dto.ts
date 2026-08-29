import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { IsIn, IsOptional } from 'class-validator';
import {
  COMPANION_MODES,
  READINESS_STATUSES,
  type CompanionMode,
  type ReadinessStatus,
} from '../companion.policy';

export class UpdateCompanionModeDto {
  @ApiProperty({ enum: COMPANION_MODES })
  @IsIn(COMPANION_MODES)
  mode!: CompanionMode;
}

export class UpdateReadinessReflectionDto {
  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  housing?: ReadinessStatus;

  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  householdAlignment?: ReadinessStatus;

  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  timeCapacity?: ReadinessStatus;

  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  financialPlan?: ReadinessStatus;

  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  supportPlan?: ReadinessStatus;

  @ApiPropertyOptional({ enum: READINESS_STATUSES })
  @IsOptional()
  @IsIn(READINESS_STATUSES)
  carePlan?: ReadinessStatus;
}
