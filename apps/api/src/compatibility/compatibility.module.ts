import { Module } from '@nestjs/common';
import { CompatibilityService } from './compatibility.service';
import { CompatibilityController } from './compatibility.controller';

@Module({
  providers: [CompatibilityService],
  controllers: [CompatibilityController],
  exports: [CompatibilityService],
})
export class CompatibilityModule {}
