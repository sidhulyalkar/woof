import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { InsightsModule } from '../insights/insights.module';
import { AdventureController } from './adventure.controller';
import { AdventureService } from './adventure.service';

@Module({
  imports: [InsightsModule, CareEventsModule],
  controllers: [AdventureController],
  providers: [AdventureService],
  exports: [AdventureService],
})
export class AdventureModule {}
