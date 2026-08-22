import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { ActivitiesController } from './activities.controller';
import { ActivitiesService } from './activities.service';

@Module({
  imports: [CareEventsModule],
  providers: [ActivitiesService],
  controllers: [ActivitiesController],
  exports: [ActivitiesService],
})
export class ActivitiesModule {}
