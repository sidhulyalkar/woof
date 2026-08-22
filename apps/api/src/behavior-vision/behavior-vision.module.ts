import { Module } from '@nestjs/common';
import { BehaviorVisionController } from './behavior-vision.controller';
import { BehaviorVisionModelService } from './behavior-vision.model';
import { BehaviorVisionService } from './behavior-vision.service';

@Module({
  controllers: [BehaviorVisionController],
  providers: [BehaviorVisionService, BehaviorVisionModelService],
  exports: [BehaviorVisionService],
})
export class BehaviorVisionModule {}
