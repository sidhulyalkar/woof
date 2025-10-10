import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { PrismaModule } from './prisma/prisma.module';
import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { PetsModule } from './pets/pets.module';
import { ActivitiesModule } from './activities/activities.module';
import { SocialModule } from './social/social.module';
import { MeetupsModule } from './meetups/meetups.module';
import { CompatibilityModule } from './compatibility/compatibility.module';
import { MeetupProposalsModule } from './meetup-proposals/meetup-proposals.module';
import { ServicesModule } from './services/services.module';
import { EventsModule } from './events/events.module';
import { GamificationModule } from './gamification/gamification.module';
import { VerificationModule } from './verification/verification.module';
import { CoActivityModule } from './co-activity/co-activity.module';
import { AnalyticsModule } from './analytics/analytics.module';
import { StorageModule } from './storage/storage.module';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    ThrottlerModule.forRoot([
      {
        name: 'short',
        ttl: 1000,
        limit: 3,
      },
      {
        name: 'medium',
        ttl: 10000,
        limit: 20,
      },
      {
        name: 'long',
        ttl: 60000,
        limit: 100,
      },
    ]),
    PrismaModule,
    AuthModule,
    UsersModule,
    PetsModule,
    ActivitiesModule,
    SocialModule,
    MeetupsModule,
    CompatibilityModule,
    MeetupProposalsModule,
    ServicesModule,
    EventsModule,
    GamificationModule,
    VerificationModule,
    CoActivityModule,
    AnalyticsModule,
    StorageModule,
    ChatModule,
  ],
  controllers: [AppController],
  providers: [
    AppService,
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
