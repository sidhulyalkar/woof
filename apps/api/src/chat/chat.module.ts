import { Module } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { JwtModule } from '@nestjs/jwt';
import { AuthModule } from '../auth/auth.module';
import { NudgesModule } from '../nudges/nudges.module';
import { PrismaModule } from '../prisma/prisma.module';
import { ChatController } from './chat.controller';
import { ChatGateway } from './chat.gateway';
import { ChatSecurityService } from './chat-security.service';
import { ChatService } from './chat.service';
import { RealtimeAdmissionService } from './realtime-admission.service';

@Module({
  imports: [
    JwtModule.registerAsync({
      inject: [ConfigService],
      useFactory: (config: ConfigService) => ({
        secret: config.get('JWT_SECRET'),
        signOptions: { expiresIn: config.get('JWT_EXPIRES_IN') || '7d' },
      }),
    }),
    AuthModule,
    NudgesModule,
    PrismaModule,
  ],
  controllers: [ChatController],
  providers: [ChatGateway, ChatSecurityService, ChatService, RealtimeAdmissionService],
})
export class ChatModule {}
