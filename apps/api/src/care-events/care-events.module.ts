import { Module } from '@nestjs/common';
import { CareEventsService } from './care-events.service';

@Module({
  providers: [CareEventsService],
  exports: [CareEventsService],
})
export class CareEventsModule {}
