import { Module } from '@nestjs/common';
import { HouseholdsModule } from '../households/households.module';
import { PetsController } from './pets.controller';
import { PetsService } from './pets.service';

@Module({
  imports: [HouseholdsModule],
  providers: [PetsService],
  controllers: [PetsController],
  exports: [PetsService],
})
export class PetsModule {}
