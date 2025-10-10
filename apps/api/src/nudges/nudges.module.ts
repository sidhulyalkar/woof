import { Module } from '@nestjs/common';
import { ScheduleModule } from '@nestjs/schedule';
import { NudgesService } from './nudges.service';
import { NudgesController } from './nudges.controller';
import { PrismaModule } from '../prisma/prisma.module';

@Module({
  imports: [ScheduleModule.forRoot(), PrismaModule],
  controllers: [NudgesController],
  providers: [NudgesService],
  exports: [NudgesService],
})
export class NudgesModule {}
