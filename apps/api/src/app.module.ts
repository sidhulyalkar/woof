import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { APP_GUARD } from '@nestjs/core';
import { ThrottlerGuard, ThrottlerModule, type ThrottlerModuleOptions } from '@nestjs/throttler';
import { ABTestModule } from './ab-testing/ab-test.module';
import { ActivitiesModule } from './activities/activities.module';
import { AnalyticsModule } from './analytics/analytics.module';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { BehaviorVisionModule } from './behavior-vision/behavior-vision.module';
import { ChatModule } from './chat/chat.module';
import { CoActivityModule } from './co-activity/co-activity.module';
import { CoachingModule } from './coaching/coaching.module';
import { CompatibilityModule } from './compatibility/compatibility.module';
import { validateEnvironment } from './config/env.validation';
import { EventsModule } from './events/events.module';
import { GamificationModule } from './gamification/gamification.module';
import { GoalsModule } from './goals/goals.module';
import { HealthLensModule } from './health-lens/health-lens.module';
import { InsightsModule } from './insights/insights.module';
import { MediaLibraryModule } from './media-library/media-library.module';
import { MeetupProposalsModule } from './meetup-proposals/meetup-proposals.module';
import { MeetupsModule } from './meetups/meetups.module';
import { MLModule } from './ml/ml.module';
import { NotificationsModule } from './notifications/notifications.module';
import { NudgesModule } from './nudges/nudges.module';
import { PetsModule } from './pets/pets.module';
import { PrismaModule } from './prisma/prisma.module';
import { PrivacyModule } from './privacy/privacy.module';
import { QuizModule } from './quiz/quiz.module';
import { ServicesModule } from './services/services.module';
import { SocialModule } from './social/social.module';
import { StorageModule } from './storage/storage.module';
import { TrustSafetyModule } from './trust-safety/trust-safety.module';
import { UsersModule } from './users/users.module';
import { VerificationModule } from './verification/verification.module';

export const throttlerOptions: ThrottlerModuleOptions = {
  skipIf: () => process.env.NODE_ENV === 'test',
  throttlers: [
    { name: 'short', ttl: 1000, limit: 3 },
    { name: 'medium', ttl: 10000, limit: 20 },
    { name: 'long', ttl: 60000, limit: 100 },
  ],
};

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
      validate: validateEnvironment,
    }),
    ThrottlerModule.forRoot(throttlerOptions),
    PrismaModule,
    PrivacyModule,
    TrustSafetyModule,
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
    CoachingModule,
    BehaviorVisionModule,
    HealthLensModule,
    MediaLibraryModule,
    AnalyticsModule,
    StorageModule,
    ChatModule,
    NudgesModule,
    NotificationsModule,
    GoalsModule,
    MLModule,
    ABTestModule,
    QuizModule,
    InsightsModule,
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
