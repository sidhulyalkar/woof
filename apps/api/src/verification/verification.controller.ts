import {
  Controller,
  Get,
  Post,
  Body,
  Patch,
  Param,
  Delete,
  UseGuards,
  Request,
  UploadedFile,
  UseInterceptors,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { ApiBearerAuth, ApiTags, ApiOperation, ApiResponse, ApiConsumes, ApiBody } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { VerificationService } from './verification.service';
import { UploadDocumentDto } from './dto/upload-document.dto';
import { UpdateVerificationDto } from './dto/update-verification.dto';

@ApiTags('verification')
@Controller('verification')
export class VerificationController {
  constructor(private readonly verificationService: VerificationService) {}

  @Post('upload')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @UseInterceptors(FileInterceptor('file'))
  @ApiConsumes('multipart/form-data')
  @ApiOperation({ summary: 'Upload a document for verification' })
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        file: { type: 'string', format: 'binary' },
        documentType: {
          type: 'string',
          enum: ['vaccination_record', 'vet_certificate', 'license', 'identity', 'other'],
        },
        petId: { type: 'string', nullable: true },
        notes: { type: 'string', nullable: true },
      },
    },
  })
  @ApiResponse({ status: 201, description: 'Document uploaded successfully' })
  async uploadDocument(
    @Request() req: AuthenticatedRequest,
    @UploadedFile() file: Express.Multer.File,
    @Body() uploadDocumentDto: UploadDocumentDto,
  ) {
    if (!file) {
      throw new BadRequestException('File is required');
    }

    const fileUrl = `/uploads/documents/${file.filename}`;
    return this.verificationService.uploadDocument(req.user.sub, fileUrl, uploadDocumentDto);
  }

  @Get('me')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get all verifications for current user' })
  @ApiResponse({ status: 200, description: 'User verifications retrieved' })
  async getMyVerifications(@Request() req: AuthenticatedRequest) {
    return this.verificationService.findAllForUser(req.user.sub);
  }

  @Get('pending')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get all pending verifications (admin only)' })
  @ApiResponse({ status: 200, description: 'Pending verifications retrieved' })
  async getPending() {
    return this.verificationService.findAllPending();
  }

  @Get('stats')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get verification statistics (admin only)' })
  @ApiResponse({ status: 200, description: 'Verification stats retrieved' })
  async getStats() {
    return this.verificationService.getStats();
  }

  @Get(':id')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Get a specific verification by ID' })
  @ApiResponse({ status: 200, description: 'Verification retrieved' })
  async findOne(@Param('id') id: string) {
    return this.verificationService.findOne(id);
  }

  @Patch(':id')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Update verification status (admin only)' })
  @ApiResponse({ status: 200, description: 'Verification status updated' })
  async updateStatus(@Param('id') id: string, @Body() updateVerificationDto: UpdateVerificationDto) {
    return this.verificationService.updateStatus(id, updateVerificationDto);
  }

  @Delete(':id')
  @ApiBearerAuth()
  @UseGuards(JwtAuthGuard)
  @ApiOperation({ summary: 'Delete a verification' })
  @ApiResponse({ status: 200, description: 'Verification deleted' })
  async remove(@Param('id') id: string, @Request() req: AuthenticatedRequest) {
    return this.verificationService.remove(id, req.user.sub);
  }
}
