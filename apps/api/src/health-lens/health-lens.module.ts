import { Module } from '@nestjs/common';
import { PrismaModule } from '../prisma/prisma.module';
import { HealthAiService } from './health-ai.service';
import { HealthLensController } from './health-lens.controller';
import { HealthLensService } from './health-lens.service';

@Module({
  imports: [PrismaModule],
  controllers: [HealthLensController],
  providers: [HealthLensService, HealthAiService],
  exports: [HealthLensService],
})
export class HealthLensModule {}
