import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { PrismaModule } from '../prisma/prisma.module';
import { CoachingController } from './coaching.controller';
import { CoachingService } from './coaching.service';

@Module({
  imports: [PrismaModule, CareEventsModule],
  controllers: [CoachingController],
  providers: [CoachingService],
  exports: [CoachingService],
})
export class CoachingModule {}
