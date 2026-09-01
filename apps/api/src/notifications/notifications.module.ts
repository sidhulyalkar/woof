import { Module } from '@nestjs/common';
import { ConnectorCryptoService } from '../connectors/connector-crypto.service';
import { PrismaModule } from '../prisma/prisma.module';
import { NotificationsController } from './notifications.controller';
import { NotificationsService } from './notifications.service';
import { PushSubscriptionStore } from './push-subscription.store';

@Module({
  imports: [PrismaModule],
  controllers: [NotificationsController],
  providers: [ConnectorCryptoService, PushSubscriptionStore, NotificationsService],
  exports: [NotificationsService],
})
export class NotificationsModule {}
