import { Module } from '@nestjs/common';
import { HouseholdsModule } from '../households/households.module';
import { CareEventsService } from './care-events.service';

@Module({
  imports: [HouseholdsModule],
  providers: [CareEventsService],
  exports: [CareEventsService],
})
export class CareEventsModule {}
