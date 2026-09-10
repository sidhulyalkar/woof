import { Module } from '@nestjs/common';
import { SocialAdventureModule } from '../social-adventure/social-adventure.module';
import { ExpeditionJournalService } from './expedition-journal.service';
import { ExpeditionsController } from './expeditions.controller';
import { ExpeditionsService } from './expeditions.service';

@Module({
  imports: [SocialAdventureModule],
  controllers: [ExpeditionsController],
  providers: [ExpeditionsService, ExpeditionJournalService],
  exports: [ExpeditionsService],
})
export class ExpeditionsModule {}
