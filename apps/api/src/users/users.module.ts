import { Module } from '@nestjs/common';
import { StorageModule } from '../storage/storage.module';
import { AccountDeletionService } from './account-deletion.service';
import { UsersController } from './users.controller';
import { UsersService } from './users.service';

@Module({
  imports: [StorageModule],
  providers: [UsersService, AccountDeletionService],
  controllers: [UsersController],
  exports: [UsersService],
})
export class UsersModule {}
