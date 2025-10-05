import { ApiProperty } from '@nestjs/swagger';
import { IsEnum, IsString, IsOptional } from 'class-validator';

export enum VerificationStatus {
  PENDING = 'pending',
  APPROVED = 'approved',
  REJECTED = 'rejected',
}

export class UpdateVerificationDto {
  @ApiProperty({ enum: VerificationStatus, description: 'Verification status' })
  @IsEnum(VerificationStatus)
  status: VerificationStatus;

  @ApiProperty({ description: 'Admin notes about the verification', required: false })
  @IsOptional()
  @IsString()
  reviewNotes?: string;
}
