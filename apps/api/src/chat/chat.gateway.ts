import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  OnGatewayConnection,
  OnGatewayDisconnect,
  ConnectedSocket,
  MessageBody,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Logger, UseGuards } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';

interface ChatMessage {
  conversationId: string;
  senderId: string;
  text: string;
  timestamp: Date;
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

  private logger = new Logger(ChatGateway.name);
  private connectedUsers = new Map<string, string>(); // socketId -> userId

  constructor(private jwtService: JwtService) {}

  async handleConnection(client: Socket) {
    try {
      // Extract JWT from handshake
      const token = client.handshake.auth.token;

      if (!token) {
        client.disconnect();
        return;
      }

      // Verify JWT
      const payload = await this.jwtService.verifyAsync(token);
      const userId = payload.sub;

      this.connectedUsers.set(client.id, userId);
      this.logger.log(`User ${userId} connected: ${client.id}`);

      // Join user's personal room
      client.join(`user:${userId}`);

      // Notify user is online
      this.server.emit('user:online', { userId });
    } catch (error) {
      this.logger.error(`Connection error: ${error.message}`);
      client.disconnect();
    }
  }

  handleDisconnect(client: Socket) {
    const userId = this.connectedUsers.get(client.id);
    if (userId) {
      this.logger.log(`User ${userId} disconnected: ${client.id}`);
      this.connectedUsers.delete(client.id);

      // Notify user is offline
      this.server.emit('user:offline', { userId });
    }
  }

  @SubscribeMessage('message:send')
  async handleMessage(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: ChatMessage,
  ) {
    const userId = this.connectedUsers.get(client.id);

    if (!userId) {
      return { error: 'Unauthorized' };
    }

    const message = {
      ...data,
      senderId: userId,
      timestamp: new Date(),
    };

    // Emit to conversation room
    this.server.to(`conversation:${data.conversationId}`).emit('message:received', message);

    return { success: true, message };
  }

  @SubscribeMessage('conversation:join')
  handleJoinConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    client.join(`conversation:${data.conversationId}`);
    this.logger.log(`User joined conversation: ${data.conversationId}`);
    return { success: true };
  }

  @SubscribeMessage('conversation:leave')
  handleLeaveConversation(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    client.leave(`conversation:${data.conversationId}`);
    this.logger.log(`User left conversation: ${data.conversationId}`);
    return { success: true };
  }

  @SubscribeMessage('typing:start')
  handleTypingStart(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    const userId = this.connectedUsers.get(client.id);
    client.to(`conversation:${data.conversationId}`).emit('typing:start', { userId });
  }

  @SubscribeMessage('typing:stop')
  handleTypingStop(
    @ConnectedSocket() client: Socket,
    @MessageBody() data: { conversationId: string },
  ) {
    const userId = this.connectedUsers.get(client.id);
    client.to(`conversation:${data.conversationId}`).emit('typing:stop', { userId });
  }
}
