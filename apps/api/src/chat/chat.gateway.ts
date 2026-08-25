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
import {
  MAX_REALTIME_PACKET_BYTES,
  parseConversationPayload,
  parseSendChatMessagePayload,
} from './chat-input-contract';
import { ChatSecurityService } from './chat-security.service';
import { RealtimeAdmissionService } from './realtime-admission.service';

const MAX_SESSION_TIMER_DELAY_MS = 2_147_483_647;

type RealtimeJwtPayload = {
  sub?: unknown;
  exp?: unknown;
};

@WebSocketGateway({
  maxHttpBufferSize: MAX_REALTIME_PACKET_BYTES,
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
  private readonly sessionExpiries = new Map<string, number>();
  private readonly sessionExpiryTimers = new Map<string, NodeJS.Timeout>();

  constructor(
    private readonly jwtService: JwtService,
    private readonly chatSecurity: ChatSecurityService,
    private readonly realtimeAdmission: RealtimeAdmissionService,
    private readonly nudgesService: NudgesService
  ) {}

  async handleConnection(client: Socket) {
    try {
      const token = client.handshake.auth?.token;
      if (typeof token !== 'string' || token.length === 0) {
        client.disconnect();
        return;
      }

      const payload = await this.jwtService.verifyAsync<RealtimeJwtPayload>(token);
      const userId = this.userIdFromPayload(payload.sub);
      const expiresAtMs = this.expiryFromPayload(payload.exp);
      if (!userId || !expiresAtMs) {
        client.disconnect();
        return;
      }

      this.connectedUsers.set(client.id, userId);
      this.sessionExpiries.set(client.id, expiresAtMs);
      this.scheduleSessionExpiry(client, expiresAtMs);
      if (!this.hasActiveSession(client, userId)) return;
      await client.join(`user:${userId}`);
    } catch (error) {
      this.clearSession(client.id);
      this.logger.warn(`Rejected socket connection: ${this.errorName(error)}`);
      client.disconnect();
    }
  }

  handleDisconnect(client: Socket) {
    this.clearSession(client.id);
  }

  @SubscribeMessage('message:send')
  async handleMessage(@ConnectedSocket() client: Socket, @MessageBody() data: unknown) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId || !this.hasActiveSession(client, userId)) {
      return { success: false, error: 'unauthorized' };
    }

    const payload = parseSendChatMessagePayload(data);
    if (!payload) return { success: false, error: 'invalid_payload' };

    const admission = this.realtimeAdmission.consume(userId, 'message');
    if (!admission.allowed) {
      return { success: false, error: 'rate_limited', retryAfterMs: admission.retryAfterMs };
    }

    try {
      const result = await this.chatSecurity.persistMessage({ userId, ...payload });

      const message = {
        id: result.message.id,
        conversationId: result.message.conversationId,
        senderId: result.message.senderId,
        text: result.message.text,
        mediaUrls: result.message.mediaUrls,
        timestamp: result.message.createdAt,
      };

      if (!result.duplicate) {
        await this.chatSecurity.withAuthorizedRealtimeRecipients(
          payload.conversationId,
          (authorizedUserIds) => {
            this.server.to(this.userRooms(authorizedUserIds)).emit('message:received', message);
          }
        );
        void this.nudgesService
          .checkChatActivityNudges(payload.conversationId, userId)
          .catch((error) => {
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
  async handleJoinConversation(@ConnectedSocket() client: Socket, @MessageBody() data: unknown) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId || !this.hasActiveSession(client, userId)) {
      return { success: false, error: 'unauthorized' };
    }

    const payload = parseConversationPayload(data);
    if (!payload) return { success: false, error: 'invalid_payload' };

    const admission = this.realtimeAdmission.consume(userId, 'membership');
    if (!admission.allowed) {
      return { success: false, error: 'rate_limited', retryAfterMs: admission.retryAfterMs };
    }

    try {
      await this.chatSecurity.assertConversationAccess(userId, payload.conversationId);
      await client.join(`conversation:${payload.conversationId}`);
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  @SubscribeMessage('conversation:leave')
  async handleLeaveConversation(@ConnectedSocket() client: Socket, @MessageBody() data: unknown) {
    const userId = this.connectedUsers.get(client.id);
    if (!userId || !this.hasActiveSession(client, userId)) {
      return { success: false, error: 'unauthorized' };
    }

    const payload = parseConversationPayload(data);
    if (!payload) return { success: false, error: 'invalid_payload' };

    const admission = this.realtimeAdmission.consume(userId, 'membership');
    if (!admission.allowed) {
      return { success: false, error: 'rate_limited', retryAfterMs: admission.retryAfterMs };
    }

    try {
      await this.chatSecurity.assertConversationAccess(userId, payload.conversationId);
      await client.leave(`conversation:${payload.conversationId}`);
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  @SubscribeMessage('typing:start')
  async handleTypingStart(@ConnectedSocket() client: Socket, @MessageBody() data: unknown) {
    return this.emitTyping(client, data, 'typing:start');
  }

  @SubscribeMessage('typing:stop')
  async handleTypingStop(@ConnectedSocket() client: Socket, @MessageBody() data: unknown) {
    return this.emitTyping(client, data, 'typing:stop');
  }

  private async emitTyping(client: Socket, data: unknown, event: 'typing:start' | 'typing:stop') {
    const userId = this.connectedUsers.get(client.id);
    if (!userId || !this.hasActiveSession(client, userId)) {
      return { success: false, error: 'unauthorized' };
    }

    const payload = parseConversationPayload(data);
    if (!payload) return { success: false, error: 'invalid_payload' };

    const admission = this.realtimeAdmission.consume(userId, 'typing');
    if (!admission.allowed) {
      return { success: false, error: 'rate_limited', retryAfterMs: admission.retryAfterMs };
    }

    try {
      await this.chatSecurity.withAuthorizedRealtimeRecipients(
        payload.conversationId,
        (authorizedUserIds) => {
          this.server
            .to(this.userRooms(authorizedUserIds))
            .except(client.id)
            .emit(event, { userId });
        },
        userId
      );
      return { success: true };
    } catch {
      return { success: false, error: 'conversation_unavailable' };
    }
  }

  private hasActiveSession(client: Socket, userId: string) {
    const expiresAtMs = this.sessionExpiries.get(client.id);
    if (!expiresAtMs || this.connectedUsers.get(client.id) !== userId) return false;

    if (expiresAtMs <= Date.now()) {
      this.expireSession(client, expiresAtMs);
      return false;
    }

    return true;
  }

  private scheduleSessionExpiry(client: Socket, expiresAtMs: number) {
    this.clearSessionTimer(client.id);

    const remainingMs = expiresAtMs - Date.now();
    if (remainingMs <= 0) {
      this.expireSession(client, expiresAtMs);
      return;
    }

    const timer = setTimeout(
      () => {
        if (this.sessionExpiries.get(client.id) !== expiresAtMs) return;

        if (Date.now() < expiresAtMs) {
          this.scheduleSessionExpiry(client, expiresAtMs);
          return;
        }

        this.expireSession(client, expiresAtMs);
      },
      Math.min(remainingMs, MAX_SESSION_TIMER_DELAY_MS)
    );
    timer.unref();
    this.sessionExpiryTimers.set(client.id, timer);
  }

  private expireSession(client: Socket, expectedExpiresAtMs: number) {
    if (this.sessionExpiries.get(client.id) !== expectedExpiresAtMs) return;

    this.clearSession(client.id);
    client.emit('session:expired', { reason: 'token_expired' });
    client.disconnect();
  }

  private clearSession(socketId: string) {
    this.connectedUsers.delete(socketId);
    this.sessionExpiries.delete(socketId);
    this.clearSessionTimer(socketId);
  }

  private clearSessionTimer(socketId: string) {
    const timer = this.sessionExpiryTimers.get(socketId);
    if (!timer) return;

    clearTimeout(timer);
    this.sessionExpiryTimers.delete(socketId);
  }

  private userIdFromPayload(value: unknown) {
    return typeof value === 'string' && value.length > 0 ? value : null;
  }

  private expiryFromPayload(value: unknown) {
    if (typeof value !== 'number' || !Number.isSafeInteger(value) || value <= 0) return null;

    const expiresAtMs = value * 1_000;
    if (!Number.isSafeInteger(expiresAtMs) || expiresAtMs <= Date.now()) return null;

    return expiresAtMs;
  }

  private userRooms(userIds: string[]) {
    return userIds.map((userId) => `user:${userId}`);
  }

  private errorName(error: unknown) {
    return error instanceof Error ? error.name : 'UnknownError';
  }
}
