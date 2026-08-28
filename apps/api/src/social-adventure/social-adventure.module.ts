import { Module } from '@nestjs/common';
import { SocialAdventureShareCandidatesController } from './social-adventure-share-candidates.controller';
import { SocialAdventureShareCandidatesService } from './social-adventure-share-candidates.service';
import { SocialAdventureController } from './social-adventure.controller';
import { SocialAdventureService } from './social-adventure.service';

@Module({
  controllers: [SocialAdventureController, SocialAdventureShareCandidatesController],
  providers: [SocialAdventureService, SocialAdventureShareCandidatesService],
  exports: [SocialAdventureService],
})
export class SocialAdventureModule {}
