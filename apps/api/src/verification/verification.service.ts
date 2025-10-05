import { Injectable, NotFoundException, BadRequestException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { UploadDocumentDto } from './dto/upload-document.dto';
import { UpdateVerificationDto, VerificationStatus } from './dto/update-verification.dto';

@Injectable()
export class VerificationService {
  constructor(private prisma: PrismaService) {}

  /**
   * Upload a document for verification
   * Note: This creates a verification record. Actual file upload would be handled by a separate upload endpoint
   * that stores the file in S3/storage and returns a fileUrl
   */
  async uploadDocument(userId: string, fileUrl: string, dto: UploadDocumentDto) {
    // Verify user exists
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new NotFoundException(`User ${userId} not found`);
    }

    // If petId is provided, verify the pet exists and belongs to this user
    if (dto.petId) {
      const pet = await this.prisma.pet.findFirst({
        where: {
          id: dto.petId,
          ownerId: userId,
        },
      });

      if (!pet) {
        throw new NotFoundException(`Pet ${dto.petId} not found or does not belong to user`);
      }
    }

    // Create verification record
    return this.prisma.verification.create({
      data: {
        userId,
        petId: dto.petId,
        documentType: dto.documentType,
        fileUrl,
        status: 'pending',
        notes: dto.notes,
      },
    });
  }

  /**
   * Get all verifications for a user
   */
  async findAllForUser(userId: string) {
    return this.prisma.verification.findMany({
      where: { userId },
      orderBy: { uploadedAt: 'desc' },
      include: {
        pet: {
          select: {
            id: true,
            name: true,
          },
        },
      },
    });
  }

  /**
   * Get a specific verification by ID
   */
  async findOne(id: string) {
    const verification = await this.prisma.verification.findUnique({
      where: { id },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            email: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
          },
        },
      },
    });

    if (!verification) {
      throw new NotFoundException(`Verification ${id} not found`);
    }

    return verification;
  }

  /**
   * Get all pending verifications (admin endpoint)
   */
  async findAllPending() {
    return this.prisma.verification.findMany({
      where: { status: 'pending' },
      orderBy: { uploadedAt: 'asc' }, // Oldest first
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            email: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
          },
        },
      },
    });
  }

  /**
   * Update verification status (admin endpoint)
   */
  async updateStatus(id: string, dto: UpdateVerificationDto) {
    const verification = await this.findOne(id);

    // Update verification status
    const updated = await this.prisma.verification.update({
      where: { id },
      data: {
        status: dto.status,
        reviewNotes: dto.reviewNotes,
        reviewedAt: new Date(),
      },
    });

    // If approved, update user's verified status
    if (dto.status === VerificationStatus.APPROVED) {
      await this.prisma.user.update({
        where: { id: verification.userId },
        data: { isVerified: true },
      });
    }

    return updated;
  }

  /**
   * Delete a verification record
   */
  async remove(id: string, userId: string) {
    const verification = await this.findOne(id);

    // Only the owner can delete their own verification
    if (verification.userId !== userId) {
      throw new BadRequestException('You can only delete your own verifications');
    }

    return this.prisma.verification.delete({
      where: { id },
    });
  }

  /**
   * Get verification statistics (admin dashboard)
   */
  async getStats() {
    const [total, pending, approved, rejected] = await Promise.all([
      this.prisma.verification.count(),
      this.prisma.verification.count({ where: { status: 'pending' } }),
      this.prisma.verification.count({ where: { status: 'approved' } }),
      this.prisma.verification.count({ where: { status: 'rejected' } }),
    ]);

    const verifiedUsers = await this.prisma.user.count({
      where: { isVerified: true },
    });

    return {
      total,
      pending,
      approved,
      rejected,
      verifiedUsers,
    };
  }
}
