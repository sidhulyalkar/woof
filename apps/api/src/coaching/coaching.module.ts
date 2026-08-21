import { Module } from '@nestjs/common';
import { PrismaModule } from '../prisma/prisma.module';
import { CoachingController } from './coaching.controller';
import { CoachingService } from './coaching.service';

@Module({
  imports: [PrismaModule],
  controllers: [CoachingController],
  providers: [CoachingService],
  exports: [CoachingService],
})
export class CoachingModule {}
