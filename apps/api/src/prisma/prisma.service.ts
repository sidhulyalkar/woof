import { Injectable, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { PrismaClient } from '@woof/database';

type DeleteManyDelegate = {
  deleteMany: () => Promise<unknown>;
};

function isDeleteManyDelegate(value: unknown): value is DeleteManyDelegate {
  if (!value || typeof value !== 'object' || !('deleteMany' in value)) return false;
  return typeof (value as { deleteMany?: unknown }).deleteMany === 'function';
}

@Injectable()
export class PrismaService extends PrismaClient implements OnModuleInit, OnModuleDestroy {
  constructor() {
    super({
      log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
    });
  }

  async onModuleInit() {
    await this.$connect();
    console.log('✅ Database connected');
  }

  async onModuleDestroy() {
    await this.$disconnect();
    console.log('👋 Database disconnected');
  }

  async cleanDatabase() {
    if (process.env.NODE_ENV === 'production') {
      throw new Error('Cannot clean database in production!');
    }

    const models = Reflect.ownKeys(this).filter(
      (key) => typeof key === 'string' && key[0] !== '_' && key[0] !== '$',
    );

    return Promise.all(
      models.map((modelKey) => {
        const delegate: unknown = Reflect.get(this, modelKey);
        return isDeleteManyDelegate(delegate) ? delegate.deleteMany() : Promise.resolve();
      }),
    );
  }
}
