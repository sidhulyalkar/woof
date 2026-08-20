import { Module } from '@nestjs/common';
import { PrivacyModule } from '../privacy/privacy.module';
import { CoActivityController } from './co-activity.controller';
import { CoActivityService } from './co-activity.service';

@Module({
  imports: [PrivacyModule],
  providers: [CoActivityService],
  controllers: [CoActivityController],
  exports: [CoActivityService],
})
export class CoActivityModule {}
