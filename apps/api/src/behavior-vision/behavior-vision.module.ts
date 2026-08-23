import { Module } from '@nestjs/common';
import { BehaviorShadowService } from './behavior-shadow.service';
import { BehaviorVisionController } from './behavior-vision.controller';
import { BehaviorVisionModelService } from './behavior-vision.model';
import { BehaviorVisionService } from './behavior-vision.service';

@Module({
  controllers: [BehaviorVisionController],
  providers: [BehaviorVisionService, BehaviorVisionModelService, BehaviorShadowService],
  exports: [BehaviorVisionService, BehaviorShadowService],
})
export class BehaviorVisionModule {}
