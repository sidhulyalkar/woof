import { Module } from '@nestjs/common';
import { HouseholdsModule } from '../households/households.module';
import { StoryController } from './story.controller';
import { StoryEnabledGuard } from './story-enabled.guard';
import { StoryService } from './story.service';

@Module({
  imports: [HouseholdsModule],
  controllers: [StoryController],
  providers: [StoryService, StoryEnabledGuard],
  exports: [StoryService],
})
export class StoryModule {}
