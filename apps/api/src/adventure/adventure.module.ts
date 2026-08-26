import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { HouseholdsModule } from '../households/households.module';
import { InsightsModule } from '../insights/insights.module';
import { AdaptiveProfileController } from './adaptive-profile.controller';
import { AdaptiveProfileService } from './adaptive-profile.service';
import { AdventureEnabledGuard } from './adventure-enabled.guard';
import { AdventureController } from './adventure.controller';
import { AdventureService } from './adventure.service';
import { PackChallengesController } from './pack-challenges.controller';
import { PackChallengesService } from './pack-challenges.service';

@Module({
  imports: [InsightsModule, CareEventsModule, HouseholdsModule],
  controllers: [AdventureController, AdaptiveProfileController, PackChallengesController],
  providers: [
    AdventureEnabledGuard,
    AdventureService,
    AdaptiveProfileService,
    PackChallengesService,
  ],
  exports: [AdventureService, AdaptiveProfileService],
})
export class AdventureModule {}
