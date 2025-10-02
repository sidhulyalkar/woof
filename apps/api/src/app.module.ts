import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
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

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: '.env',
    }),
    PrismaModule,
    AuthModule,
    UsersModule,
    PetsModule,
    ActivitiesModule,
    SocialModule,
    MeetupsModule,
    CompatibilityModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
