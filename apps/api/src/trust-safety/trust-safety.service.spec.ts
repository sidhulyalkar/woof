import { TrustSafetyService } from './trust-safety.service';

describe('TrustSafetyService relationship serialization', () => {
  function build() {
    const events: string[] = [];
    const block = {
      id: 'block-1',
      userId: 'user-a',
      blockedId: 'user-b',
      createdAt: new Date('2026-08-23T20:00:00.000Z'),
    };
    const tx = {
      $queryRaw: jest.fn().mockImplementation(async () => {
        events.push('lock');
        return [{ locked: 1 }];
      }),
      blockedUser: {
        upsert: jest.fn().mockImplementation(async () => {
          events.push('block-upsert');
          return block;
        }),
        deleteMany: jest.fn().mockImplementation(async () => {
          events.push('block-delete');
          return { count: 1 };
        }),
      },
    };
    const prisma = {
      user: {
        findUnique: jest.fn().mockResolvedValue({ id: 'user-b' }),
        findMany: jest.fn(),
      },
      $transaction: jest
        .fn()
        .mockImplementation(async (callback: (client: typeof tx) => unknown) => {
          events.push('transaction:start');
          const result = await callback(tx);
          events.push('transaction:commit');
          return result;
        }),
      meetupProposal: {
        updateMany: jest.fn().mockImplementation(async () => {
          events.push('meetup-cleanup');
          return { count: 0 };
        }),
      },
      pet: {
        findMany: jest.fn().mockImplementation(async () => {
          events.push('pet-read');
          return [];
        }),
      },
      petEdge: { updateMany: jest.fn() },
      telemetry: {
        create: jest.fn().mockImplementation(async () => {
          events.push('telemetry');
          return { id: 'telemetry-1' };
        }),
      },
      blockedUser: {
        findMany: jest.fn(),
        findFirst: jest.fn(),
      },
      reportFlag: { create: jest.fn(), findMany: jest.fn() },
    };

    return { events, prisma, tx, service: new TrustSafetyService(prisma as never) };
  }

  it('commits the block behind the shared relationship lock before ancillary cleanup', async () => {
    const { events, prisma, tx, service } = build();

    await expect(
      service.blockUser('user-a', { blockedUserId: 'user-b', reason: 'safety boundary' } as never)
    ).resolves.toEqual({
      id: 'block-1',
      blockedUserId: 'user-b',
      createdAt: new Date('2026-08-23T20:00:00.000Z'),
    });

    expect(prisma.$transaction).toHaveBeenCalledTimes(1);
    expect(tx.$queryRaw).toHaveBeenCalledTimes(1);
    expect(tx.blockedUser.upsert).toHaveBeenCalledTimes(1);
    expect(events.indexOf('lock')).toBeLessThan(events.indexOf('block-upsert'));
    expect(events.indexOf('block-upsert')).toBeLessThan(events.indexOf('transaction:commit'));
    expect(events.indexOf('transaction:commit')).toBeLessThan(events.indexOf('meetup-cleanup'));
    expect(events.indexOf('transaction:commit')).toBeLessThan(events.indexOf('telemetry'));
  });

  it('serializes unblock through the same relationship lock', async () => {
    const { events, tx, service } = build();

    await expect(service.unblockUser('user-a', 'user-b')).resolves.toEqual({ unblocked: true });

    expect(events.indexOf('lock')).toBeLessThan(events.indexOf('block-delete'));
    expect(tx.blockedUser.deleteMany).toHaveBeenCalledWith({
      where: { userId: 'user-a', blockedId: 'user-b' },
    });
  });
});
