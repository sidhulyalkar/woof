import { Module } from '@nestjs/common';
import { HouseholdsModule } from '../households/households.module';
import { SocialController } from './social.controller';
import { SocialService } from './social.service';

@Module({
  imports: [HouseholdsModule],
  providers: [SocialService],
  controllers: [SocialController],
  exports: [SocialService],
})
export class SocialModule {}
