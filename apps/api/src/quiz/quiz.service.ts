import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { SaveQuizResponseDto } from './dto/save-quiz-response.dto';

@Injectable()
export class QuizService {
  constructor(private readonly prisma: PrismaService) {}

  async save(userId: string, dto: SaveQuizResponseDto) {
    if (dto.petId) {
      const ownedPet = await this.prisma.pet.findFirst({
        where: { id: dto.petId, ownerId: userId },
        select: { id: true },
      });

      if (!ownedPet) {
        throw new NotFoundException('Pet not found');
      }
    }

    const timestamp = new Date().toISOString();
    const responses = Object.entries(dto.responses).map(([questionId, answer]) => ({
      questionId,
      answer,
      timestamp,
    }));

    return this.prisma.quizResponse.create({
      data: {
        userId,
        petId: dto.petId,
        sessionId: dto.sessionId,
        responses: responses as Prisma.InputJsonValue,
        completedAt: new Date(),
      },
      select: {
        id: true,
        petId: true,
        sessionId: true,
        completedAt: true,
      },
    });
  }

  async latest(userId: string) {
    return this.prisma.quizResponse.findFirst({
      where: { userId },
      orderBy: { completedAt: 'desc' },
      select: {
        id: true,
        petId: true,
        sessionId: true,
        responses: true,
        completedAt: true,
      },
    });
  }
}
