import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { HouseholdsModule } from '../households/households.module';
import { DailySignalsService } from './daily-signals.service';
import { IntelligenceController } from './intelligence.controller';
import { IntelligenceProjectionService } from './intelligence-projection.service';

@Module({
  imports: [HouseholdsModule, CareEventsModule],
  controllers: [IntelligenceController],
  providers: [IntelligenceProjectionService, DailySignalsService],
  exports: [IntelligenceProjectionService, DailySignalsService],
})
export class IntelligenceModule {}
