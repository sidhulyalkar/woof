import { Module } from '@nestjs/common';
import { MLModule } from '../ml/ml.module';
import { CompatibilityController } from './compatibility.controller';
import { CompatibilityService } from './compatibility.service';

@Module({
  imports: [MLModule],
  providers: [CompatibilityService],
  controllers: [CompatibilityController],
  exports: [CompatibilityService],
})
export class CompatibilityModule {}
