import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { HouseholdsModule } from '../households/households.module';
import { ActivitiesController } from './activities.controller';
import { ActivitiesService } from './activities.service';

@Module({
  imports: [CareEventsModule, HouseholdsModule],
  providers: [ActivitiesService],
  controllers: [ActivitiesController],
  exports: [ActivitiesService],
})
export class ActivitiesModule {}
