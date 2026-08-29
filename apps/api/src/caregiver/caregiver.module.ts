import { Module } from '@nestjs/common';
import { TrustSafetyModule } from '../trust-safety/trust-safety.module';
import { CaregiverController } from './caregiver.controller';
import { CaregiverOperationalStore } from './caregiver-operational.store';
import { CaregiverService } from './caregiver.service';
import { PetCapabilityAuthority } from './pet-capability-authority';

@Module({
  imports: [TrustSafetyModule],
  controllers: [CaregiverController],
  providers: [CaregiverOperationalStore, CaregiverService, PetCapabilityAuthority],
  exports: [CaregiverService, PetCapabilityAuthority],
})
export class CaregiverModule {}
