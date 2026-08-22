import { Module } from '@nestjs/common';
import { CareEventsModule } from '../care-events/care-events.module';
import { HouseholdsModule } from '../households/households.module';
import { NotificationsModule } from '../notifications/notifications.module';
import { AutopilotEnabledGuard } from './autopilot-enabled.guard';
import { AutopilotController } from './autopilot.controller';
import { AutopilotService } from './autopilot.service';

@Module({
  imports: [CareEventsModule, HouseholdsModule, NotificationsModule],
  controllers: [AutopilotController],
  providers: [AutopilotEnabledGuard, AutopilotService],
  exports: [AutopilotService],
})
export class AutopilotModule {}
