import { Module } from '@nestjs/common';
import { PrismaModule } from '../prisma/prisma.module';
import { StorageModule } from '../storage/storage.module';
import { MediaDerivativeWorkerService } from './media-derivative.worker';
import { MediaLibraryController } from './media-library.controller';
import { MediaLibraryService } from './media-library.service';

@Module({
  imports: [PrismaModule, StorageModule],
  controllers: [MediaLibraryController],
  providers: [MediaLibraryService, MediaDerivativeWorkerService],
  exports: [MediaLibraryService],
})
export class MediaLibraryModule {}
