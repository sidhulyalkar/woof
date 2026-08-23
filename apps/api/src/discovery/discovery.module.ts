import { Module } from '@nestjs/common';
import { PrismaModule } from '../prisma/prisma.module';
import { DiscoveryController } from './discovery.controller';
import { DiscoveryLocationStore } from './discovery-location.store';
import { DiscoveryService } from './discovery.service';

@Module({
  imports: [PrismaModule],
  controllers: [DiscoveryController],
  providers: [DiscoveryLocationStore, DiscoveryService],
  exports: [DiscoveryService],
})
export class DiscoveryModule {}
