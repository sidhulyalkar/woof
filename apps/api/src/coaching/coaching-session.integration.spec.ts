import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CoachingService } from './coaching.service';
import { RecordTrainingSessionDto } from './dto/coaching.dto';

describe('coaching session persistence', () => {
  const prisma = new PrismaService();
  const service = new CoachingService(prisma);
  const owners: string[] = [];
  beforeAll(() => prisma.$connect());
  afterAll(async () => {
    await prisma.user.deleteMany({ where: { id: { in: owners } } });
    await prisma.$disconnect();
  });
  async function fixture() {
    const suffix = randomUUID();
    const user = await prisma.user.create({
      data: { handle: suffix, email: `${suffix}@example.test` },
    });
    owners.push(user.id);
    const pet = await prisma.pet.create({
      data: { ownerId: user.id, name: 'Practice', species: 'DOG' },
    });
    const plan = (await service.createPlan(user.id, {
      petId: pet.id,
      templateId: 'dog-check-in',
    }))!;
    const input: RecordTrainingSessionDto = {
      sessionKey: randomUUID(),
      expectedPlanUpdatedAt: plan.updatedAt,
      attempts: 5,
      successes: 5,
      durationSeconds: 40,
      distractionLevel: 1,
      rewardType: 'food',
    };
    return { user, pet, plan, input };
  }
  it('converges simultaneous exact retries into one activity and plan increment', async () => {
    const { user, plan, input } = await fixture();
    const results = await Promise.all(
      Array.from({ length: 4 }, () => service.recordSession(user.id, plan.id, input))
    );
    expect(new Set(results.map((r) => r.activityId)).size).toBe(1);
    results.forEach((result) => expect(result).toEqual(results[0]));
    expect(await prisma.activity.count({ where: { userId: user.id } })).toBe(1);
    expect(
      (await prisma.mutualGoal.findUniqueOrThrow({ where: { id: plan.id } })).currentValue
    ).toBe(1);
  });
  it('rejects conflicting retries and replays the original receipt after pausing', async () => {
    const { user, plan, input } = await fixture();
    const receipt = await service.recordSession(user.id, plan.id, input);
    await expect(
      service.recordSession(user.id, plan.id, { ...input, successes: 3 })
    ).rejects.toMatchObject({ status: 409 });
    await service.setPlanStatus(user.id, plan.id, { status: 'PAUSED' });
    expect(await service.recordSession(user.id, plan.id, input)).toEqual(receipt);
  });
  it('rejects a different concurrent session based on the stale plan version', async () => {
    const { user, plan, input } = await fixture();
    const results = await Promise.allSettled([
      service.recordSession(user.id, plan.id, input),
      service.recordSession(user.id, plan.id, { ...input, sessionKey: randomUUID() }),
    ]);
    expect(results.filter((r) => r.status === 'fulfilled')).toHaveLength(1);
    expect(results.filter((r) => r.status === 'rejected')).toHaveLength(1);
    expect(await prisma.activity.count({ where: { userId: user.id } })).toBe(1);
  });
  it('requires two successes at each level and reads the applied decision', async () => {
    const { user, pet, plan, input } = await fixture();
    let version = plan.updatedAt;
    const levels: number[] = [];
    for (let index = 0; index < 4; index++) {
      const result = await service.recordSession(user.id, plan.id, {
        ...input,
        sessionKey: randomUUID(),
        expectedPlanUpdatedAt: version,
      });
      levels.push(result.plan!.level);
      version = result.plan!.updatedAt;
      expect((await service.getDashboard(user.id, pet.id)).activePlan?.nextPractice).toEqual(
        result.decision
      );
    }
    expect(levels).toEqual([1, 2, 2, 3]);
  });
  it('reauthorizes replay after pet ownership changes', async () => {
    const owner = await fixture();
    const other = await fixture();
    await service.recordSession(owner.user.id, owner.plan.id, owner.input);
    await prisma.pet.update({ where: { id: owner.pet.id }, data: { ownerId: other.user.id } });
    await expect(
      service.recordSession(owner.user.id, owner.plan.id, owner.input)
    ).rejects.toMatchObject({ status: 403 });
  });
  it('rolls back both activity and plan when receipt storage fails', async () => {
    const { user, plan, input } = await fixture();
    const failingPrisma = {} as PrismaService;
    failingPrisma.$transaction = ((callback: (tx: Prisma.TransactionClient) => Promise<unknown>) =>
      prisma.$transaction((tx) =>
        callback(
          new Proxy(tx, {
            get(target, key) {
              if (key === '$executeRaw')
                return () => {
                  throw new Error('receipt storage unavailable');
                };
              return Reflect.get(target, key);
            },
          })
        )
      )) as PrismaService['$transaction'];
    await expect(
      new CoachingService(failingPrisma).recordSession(user.id, plan.id, input)
    ).rejects.toThrow('receipt storage unavailable');
    expect(await prisma.activity.count({ where: { userId: user.id } })).toBe(0);
    expect(
      (await prisma.mutualGoal.findUniqueOrThrow({ where: { id: plan.id } })).currentValue
    ).toBe(0);
    await expect(service.recordSession(user.id, plan.id, input)).resolves.toHaveProperty(
      'activityId'
    );
  });
  it('supports a safe stop before the first repetition', async () => {
    const { user, plan, input } = await fixture();
    const receipt = await service.recordSession(user.id, plan.id, {
      ...input,
      attempts: 0,
      successes: 0,
      stoppedEarly: true,
    });
    expect(receipt.decision.action).toBe('decrease');
  });
});
