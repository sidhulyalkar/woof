import { Module } from '@nestjs/common';
import { HouseholdsModule } from '../households/households.module';
import { IntelligenceProjectionService } from './intelligence-projection.service';

@Module({
  imports: [HouseholdsModule],
  providers: [IntelligenceProjectionService],
  exports: [IntelligenceProjectionService],
})
export class IntelligenceModule {}
