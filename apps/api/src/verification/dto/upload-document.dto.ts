import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsEnum, IsOptional } from 'class-validator';

export enum DocumentType {
  VACCINATION_RECORD = 'vaccination_record',
  VET_CERTIFICATE = 'vet_certificate',
  LICENSE = 'license',
  IDENTITY = 'identity',
  OTHER = 'other',
}

export class UploadDocumentDto {
  @ApiProperty({ enum: DocumentType, description: 'Type of document' })
  @IsEnum(DocumentType)
  documentType: DocumentType;

  @ApiProperty({ description: 'Pet ID (if document is pet-specific)', required: false })
  @IsOptional()
  @IsString()
  petId?: string;

  @ApiProperty({ description: 'Additional notes about the document', required: false })
  @IsOptional()
  @IsString()
  notes?: string;
}
