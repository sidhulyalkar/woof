import { ConfigService } from '@nestjs/config';
import { prisma } from '@woof/database';
import { ConnectorCryptoService } from '../src/connectors/connector-crypto.service';
import { PushSubscriptionStore } from '../src/notifications/push-subscription.store';
import { PrismaService } from '../src/prisma/prisma.service';

async function main() {
  const batchSize = process.env.PUSH_SUBSCRIPTION_MIGRATION_BATCH_SIZE
    ? Number(process.env.PUSH_SUBSCRIPTION_MIGRATION_BATCH_SIZE)
    : 100;
  const config = new ConfigService({
    CONNECTOR_CREDENTIALS_KEY: process.env.CONNECTOR_CREDENTIALS_KEY,
  });
  const crypto = new ConnectorCryptoService(config);
  const store = new PushSubscriptionStore(prisma as unknown as PrismaService, crypto, config);
  const report = await store.migrateLegacyRows(batchSize);

  process.stdout.write(
    `${JSON.stringify({
      schemaVersion: 1,
      migration: 'web_push_subscription_encryption_v1',
      ...report,
    })}\n`
  );
}

void main()
  .catch(() => {
    process.stderr.write('Push subscription encryption migration failed\n');
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
