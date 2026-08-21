import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { InsightsModule } from '../insights/insights.module';
import { AdventureController } from './adventure.controller';
import { AdventureService } from './adventure.service';
import { PackChallengesController } from './pack-challenges.controller';
import { PackChallengesService } from './pack-challenges.service';

@Module({
  imports: [InsightsModule, CareEventsModule],
  controllers: [AdventureController, PackChallengesController],
  providers: [AdventureService, PackChallengesService],
  exports: [AdventureService],
})
export class AdventureModule {}
