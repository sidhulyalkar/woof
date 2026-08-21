import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  ConnectedSocket,
  MessageBody,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Logger, UseGuards } from '@nestjs/common';
import { Server, Socket } from 'socket.io';
import { JwtService } from '@nestjs/jwt';
import { PrismaService } from '../prisma/prisma.service';
import { WsJwtGuard } from '../auth/guards/ws-jwt.guard';
import { NudgesService } from '../nudges/nudges.service';

interface AuthenticatedSocket extends Socket {
  userId?: string;
}

@WebSocketGateway({
  cors: {
    origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
    credentials: true,
  },
  namespace: '/chat',
})
export class ChatGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(ChatGateway.name);

  constructor(
    private readonly jwtService: JwtService,
    private readonly prisma: PrismaService,
    private readonly nudgesService: NudgesService
  ) {}

  async handleConnection(client: AuthenticatedSocket) {
    try {
      const token =
        client.handshake.auth?.token || client.handshake.headers?.authorization?.split(' ')[1];

      if (!token) {
        client.disconnect();
        return;
      }

      const payload = this.jwtService.verify(token);
      client.userId = payload.sub;
      client.join(`user:${payload.sub}`);

      this.logger.log(`Client connected: ${client.id} (User: ${payload.sub})`);
    } catch (error) {
      this.logger.error(`Authentication failed: ${error.message}`);
      client.disconnect();
    }
  }

  handleDisconnect(client: AuthenticatedSocket) {
    this.logger.log(`Client disconnected: ${client.id}`);
  }

  @UseGuards(WsJwtGuard)
  @SubscribeMessage('message:send')
  async handleMessage(
    @ConnectedSocket() client: AuthenticatedSocket,
    @MessageBody()
    data: {
      conversationId: string;
      text: string;
    }
  ) {
    const userId = client.userId;
    if (!userId) {
      return { success: false, error: 'Unauthorized' };
    }

    const participant = await this.prisma.conversationParticipant.findFirst({
      where: {
        conversationId: data.conversationId,
        userId,
      },
      select: { id: true },
    });

    if (!participant) {
      return { success: false, error: 'Conversation not found' };
    }

    const message = {
      ...data,
      senderId: userId,
      timestamp: new Date(),
    };

    try {
      await this.prisma.message.create({
        data: {
          conversationId: data.conversationId,
          senderId: userId,
          text: data.text,
        },
      });

      await this.nudgesService
        .checkChatActivityNudges(data.conversationId, userId)
        .catch((err) => {
          this.logger.error(`Failed to check chat nudges: ${err.message}`);
        });
    } catch (error) {
      this.logger.error(`Failed to save message: ${error.message}`);
    }

    this.server.to(`conversation:${data.conversationId}`).emit('message:received', message);

    return { success: true, message };
  }

  @SubscribeMessage('conversation:join')
  handleJoinConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string }
  ) {
    client.join(`conversation:${data.conversationId}`);
    return { success: true };
  }

  @SubscribeMessage('conversation:leave')
  handleLeaveConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string }
  ) {
    client.leave(`conversation:${data.conversationId}`);
    return { success: true };
  }

  @SubscribeMessage('typing:start')
  handleTypingStart(
    @ConnectedSocket() client: AuthenticatedSocket,
    @MessageBody() data: { conversationId: string }
  ) {
    client.to(`conversation:${data.conversationId}`).emit('typing:started', {
      userId: client.userId,
      conversationId: data.conversationId,
    });
  }

  @SubscribeMessage('typing:stop')
  handleTypingStop(
    @ConnectedSocket() client: AuthenticatedSocket,
    @MessageBody() data: { conversationId: string }
  ) {
    client.to(`conversation:${data.conversationId}`).emit('typing:stopped', {
      userId: client.userId,
      conversationId: data.conversationId,
    });
  }
}
