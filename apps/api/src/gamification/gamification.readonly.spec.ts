import { GamificationService } from './gamification.service';

function buildService() {
  const prisma = {
    user: {
      findUnique: jest.fn(),
      update: jest.fn(),
    },
    badgeAward: {
      findMany: jest.fn(),
      create: jest.fn(),
    },
    weeklyStreak: {
      findUnique: jest.fn(),
      update: jest.fn(),
      create: jest.fn(),
    },
  };
  return {
    prisma,
    service: new GamificationService(prisma as never),
  };
}

describe('GamificationService historical compatibility', () => {
  it('normalizes an expired streak in memory without mutating legacy storage', async () => {
    const { prisma, service } = buildService();
    prisma.weeklyStreak.findUnique.mockResolvedValue({
      userId: 'user-1',
      currentWeek: 9,
      lastActivityAt: new Date('2020-01-01T00:00:00.000Z'),
    });

    const result = await service.getUserStreak('user-1');

    expect(result).toMatchObject({ currentWeek: 0 });
    expect(prisma.weeklyStreak.update).not.toHaveBeenCalled();
    expect(prisma.weeklyStreak.create).not.toHaveBeenCalled();
  });

  it('returns historical totals without writing them', async () => {
    const { prisma, service } = buildService();
    prisma.user.findUnique.mockResolvedValue({ totalPoints: 42 });

    await expect(service.getUserPoints('user-1')).resolves.toEqual({ totalPoints: 42 });
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it('returns historical badges without creating new awards', async () => {
    const { prisma, service } = buildService();
    prisma.badgeAward.findMany.mockResolvedValue([{ badgeType: 'EARLY_ADOPTER' }]);

    await expect(service.getUserBadges('user-1')).resolves.toEqual([
      { badgeType: 'EARLY_ADOPTER' },
    ]);
    expect(prisma.badgeAward.create).not.toHaveBeenCalled();
  });
});
