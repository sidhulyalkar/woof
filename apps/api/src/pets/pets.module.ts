import { Module } from '@nestjs/common';
import { PetsService } from './pets.service';
import { PetsController } from './pets.controller';

@Module({
  providers: [PetsService],
  controllers: [PetsController],
  exports: [PetsService],
})
export class PetsModule {}
