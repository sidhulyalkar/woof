import { Module } from '@nestjs/common';
import { SocialAdventureController } from './social-adventure.controller';
import { SocialAdventureService } from './social-adventure.service';

@Module({
  controllers: [SocialAdventureController],
  providers: [SocialAdventureService],
  exports: [SocialAdventureService],
})
export class SocialAdventureModule {}
