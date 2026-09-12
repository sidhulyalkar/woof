import { Module } from '@nestjs/common';
import { PackAccessService } from './pack-access.service';
import { PackLocalityService } from './pack-locality.service';
import { SocialAdventureShareCandidatesController } from './social-adventure-share-candidates.controller';
import { SocialAdventureShareCandidatesService } from './social-adventure-share-candidates.service';
import { SocialAdventureController } from './social-adventure.controller';
import { SocialAdventureService } from './social-adventure.service';

@Module({
  controllers: [SocialAdventureController, SocialAdventureShareCandidatesController],
  providers: [
    PackAccessService,
    PackLocalityService,
    SocialAdventureService,
    SocialAdventureShareCandidatesService,
  ],
  exports: [PackAccessService, PackLocalityService, SocialAdventureService],
})
export class SocialAdventureModule {}
