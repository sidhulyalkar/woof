import { Module } from '@nestjs/common';
import { AdventureModule } from '../adventure/adventure.module';
import { AutopilotModule } from '../autopilot/autopilot.module';
import { CareEventsModule } from '../care-events/care-events.module';
import { ConnectorsModule } from '../connectors/connectors.module';
import { ConciergeController } from './concierge.controller';
import { ConciergeEnabledGuard } from './concierge-enabled.guard';
import { ConciergeService } from './concierge.service';

@Module({
  imports: [AdventureModule, AutopilotModule, CareEventsModule, ConnectorsModule],
  controllers: [ConciergeController],
  providers: [ConciergeEnabledGuard, ConciergeService],
  exports: [ConciergeService],
})
export class ConciergeModule {}
