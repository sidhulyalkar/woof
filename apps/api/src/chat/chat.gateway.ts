import { Logger } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import {
  ConnectedSocket,
  MessageBody,
  OnGatewayConnection,
  OnGatewayDisconnect,
  SubscribeMessage,
  WebSocketGateway,
  WebSocketServer,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { NudgesService } from '../nudges/nudges.service';
import { ChatSecurityService } from './chat-security.service';

interface SendChatMessage {
  conversationId: string;
  clientMessageId: string;
  text: string;
}

@WebSocketGateway({
  cors: {
    origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:3000'],
    credentials: true,
  },
})
export class ChatGateway implements OnGatewayConnection, OnGatewayDisconnect {
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(ChatGateway.name);
  private readonly connectedUsers = new Map<string, string>();

  constructor(
    private readonly jwtService: JwtService,
    private readonly chatSecurity: ChatSecurityService,
    private readonly nudgesService: NudgesService,
  ) {}

  async handleConnection(client: Socket) {
    try {
      const token = client.handshake.auth.token;
      if (!token) {
        client.disconnect();
        return;
      }

      const payload = await this.jwtService.verifyAsync<{ sub?: string }>(token);
      if (!payload.sub) {
        client.disconnect();
        return;
      }

      this.connectedUsers.set(client.id, payload.sub);
      await client.join(`user:${payload.sub}`);
    } catch (error) {
      this.logger.warn(`Rejected socket connection: ${this.errorName(error)}`);
      client.disconnect();
    }
  }

  handleDisconnect(client: Socket) {
    this.connectedUsers.delete(client.id);
  }

  @SubscribeMessage('message:send')
  async handleMessage(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: SendChatMessage,
  ) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId) return { success: false, error: 'unauthorized' };

    try {
      const result = await this.chatSecurity.persistMessage({
        userId,
        conversationId: data.conversationId,
        clientMessageId: data.clientMessageId,
        text: data.text,
      });

      const message = {
        id: result.message.id,
        conversationId: result.message.conversationId,
        senderId: result.message.senderId,
        text: result.message.text,
        mediaUrls: result.message.mediaUrls,
        timestamp: result.message.createdAt,
      };

      if (!result.duplicate) {
        this.server.to(`conversation:${data.conversationId}`).emit('message:received', message);
        void this.nudgesService.checkChatActivityNudges(data.conversationId, userId).catch((error) => {
          this.logger.warn(`Chat nudge check failed: ${this.errorName(error)}`);
        });
      }

      return { success: true, duplicate: result.duplicate, message };
    } catch (error) {
      this.logger.warn(`Rejected chat message: ${this.errorName(error)}`);
      return { success: false, error: 'message_rejected' };
    }
  }

  @SubscribeMessage('conversation:join')
  async handleJoinConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId) return { success: false, error: 'unauthorized' };

    try {
      await this.chatSecurity.assertConversationAccess(userId, data.conversationId);
      await client.join(`conversation:${data.conversationId}`);
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  @SubscribeMessage('conversation:leave')
  async handleLeaveConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId) return { success: false, error: 'unauthorized' };

    try {
      await this.chatSecurity.assertConversationAccess(userId, data.conversationId);
      await client.leave(`conversation:${data.conversationId}`);
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  @SubscribeMessage('typing:start')
  async handleTypingStart(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    return this.emitTyping(client, data.conversationId, 'typing:start');
  }

  @SubscribeMessage('typing:stop')
  async handleTypingStop(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    return this.emitTyping(client, data.conversationId, 'typing:stop');
  }

  private async emitTyping(client: Socket, conversationId: string, event: 'typing:start' | 'typing:stop') {
    const userId = this.connectedUsers.get(client.id);
    if (!userId) return { success: false, error: 'unauthorized' };

    try {
      await this.chatSecurity.assertConversationAccess(userId, conversationId);
      client.to(`conversation:${conversationId}`).emit(event, { userId });
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  private errorName(error: unknown) {
    return error instanceof Error ? error.name : 'UnknownError';
  }
}
