import { Injectable, NotFoundException, ForbiddenException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';
import { CreateGoalDto, UpdateGoalDto, GoalStatus } from './dto';

@Injectable()
export class GoalsService {
  constructor(private prisma: PrismaService) {}

  async create(userId: string, createGoalDto: CreateGoalDto) {
    const { petId, ...goalData } = createGoalDto;

    // Verify pet belongs to user
    const pet = await this.prisma.pet.findUnique({
      where: { id: petId },
    });

    if (!pet || pet.ownerId !== userId) {
      throw new ForbiddenException('You do not have access to this pet');
    }

    return this.prisma.mutualGoal.create({
      data: {
        userId,
        petId,
        goalType: goalData.goalType,
        period: goalData.period,
        targetNumber: goalData.targetNumber,
        targetUnit: goalData.targetUnit,
        startDate: new Date(goalData.startDate),
        endDate: new Date(goalData.endDate),
        reminderTime: goalData.reminderTime,
        isRecurring: goalData.isRecurring || false,
        metadata: goalData.metadata || {},
        currentValue: 0,
        progress: 0,
        status: 'ACTIVE',
      },
      include: {
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async findAll(userId: string, petId?: string, status?: string) {
    const where: Prisma.MutualGoalWhereInput = { userId };

    if (petId) {
      where.petId = petId;
    }

    if (status) {
      where.status = status;
    }

    return this.prisma.mutualGoal.findMany({
      where,
      include: {
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  async findOne(userId: string, id: string) {
    const goal = await this.prisma.mutualGoal.findUnique({
      where: { id },
      include: {
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });

    if (!goal) {
      throw new NotFoundException('Goal not found');
    }

    if (goal.userId !== userId) {
      throw new ForbiddenException('You do not have access to this goal');
    }

    return goal;
  }

  async update(userId: string, id: string, updateGoalDto: UpdateGoalDto) {
    // Verify goal belongs to user
    const goal = await this.findOne(userId, id);

    return this.prisma.mutualGoal.update({
      where: { id },
      data: updateGoalDto,
      include: {
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async remove(userId: string, id: string) {
    // Verify goal belongs to user
    await this.findOne(userId, id);

    await this.prisma.mutualGoal.delete({
      where: { id },
    });

    return { message: 'Goal deleted successfully' };
  }

  /**
   * Update goal progress based on activities
   */
  async updateProgress(userId: string, goalId: string, value: number) {
    const goal = await this.findOne(userId, goalId);

    const currentValue = goal.currentValue + value;
    const progress = Math.min((currentValue / goal.targetNumber) * 100, 100);
    const status = progress >= 100 ? GoalStatus.COMPLETED : goal.status;

    // Update streak if daily goal and not already completed today
    const today = new Date().toISOString().split('T')[0];
    const completedDays = (goal.completedDays as string[]) || [];
    const updatedCompletedDays = completedDays.includes(today)
      ? completedDays
      : [...completedDays, today];

    const streakCount =
      status === GoalStatus.COMPLETED && !completedDays.includes(today)
        ? goal.streakCount + 1
        : goal.streakCount;

    const bestStreak = Math.max(streakCount, goal.bestStreak);

    return this.prisma.mutualGoal.update({
      where: { id: goalId },
      data: {
        currentValue,
        progress,
        status,
        streakCount,
        bestStreak,
        completedDays: updatedCompletedDays,
      },
      include: {
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  /**
   * Get goal statistics for a user
   */
  async getStatistics(userId: string) {
    const goals = await this.prisma.mutualGoal.findMany({
      where: { userId },
    });

    const activeGoals = goals.filter((g) => g.status === 'ACTIVE').length;
    const completedGoals = goals.filter((g) => g.status === 'COMPLETED').length;
    const failedGoals = goals.filter((g) => g.status === 'FAILED').length;
    const totalProgress =
      goals.length > 0
        ? goals.reduce((sum, g) => sum + g.progress, 0) / goals.length
        : 0;
    const longestStreak = Math.max(...goals.map((g) => g.bestStreak), 0);
    const currentStreak = Math.max(
      ...goals.filter((g) => g.status === 'ACTIVE').map((g) => g.streakCount),
      0,
    );

    return {
      totalGoals: goals.length,
      activeGoals,
      completedGoals,
      failedGoals,
      averageProgress: Math.round(totalProgress),
      longestStreak,
      currentStreak,
    };
  }
}
