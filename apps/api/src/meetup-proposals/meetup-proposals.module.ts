import { Module } from '@nestjs/common';
import { MeetupProposalsService } from './meetup-proposals.service';
import { MeetupProposalsController } from './meetup-proposals.controller';

@Module({
  providers: [MeetupProposalsService],
  controllers: [MeetupProposalsController],
  exports: [MeetupProposalsService],
})
export class MeetupProposalsModule {}
