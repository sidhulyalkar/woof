import { Module } from '@nestjs/common';
import { CoActivityService } from './co-activity.service';
import { CoActivityController } from './co-activity.controller';

@Module({
  providers: [CoActivityService],
  controllers: [CoActivityController],
  exports: [CoActivityService],
})
export class CoActivityModule {}
