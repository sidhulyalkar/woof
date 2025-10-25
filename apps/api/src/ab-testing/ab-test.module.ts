import { Module } from '@nestjs/common';
import { ABTestService } from './ab-test.service';
import { ABTestController } from './ab-test.controller';

@Module({
  providers: [ABTestService],
  controllers: [ABTestController],
  exports: [ABTestService],
})
export class ABTestModule {}
