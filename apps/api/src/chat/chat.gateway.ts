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
import type { Prisma } from '@woof/database';
import { Server, Socket } from 'socket.io';
import { SessionAuthorityService } from '../auth/session-authority.service';
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
  sid?: unknown;
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
  private readonly connectedSessions = new Map<string, string>();
  private readonly sessionExpiries = new Map<string, number>();
  private readonly sessionExpiryTimers = new Map<string, NodeJS.Timeout>();

  constructor(
    private readonly jwtService: JwtService,
    private readonly sessionAuthority: SessionAuthorityService,
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
      const userId = this.stringClaim(payload.sub);
      const sessionId = this.stringClaim(payload.sid);
      const expiresAtMs = this.expiryFromPayload(payload.exp);
      if (!userId || !sessionId || !expiresAtMs) {
        client.disconnect();
        return;
      }

      const admission = await this.sessionAuthority.withActiveSession(
        sessionId,
        userId,
        async () => {
          if (this.isSocketDisconnected(client)) return false;

          this.connectedUsers.set(client.id, userId);
          this.connectedSessions.set(client.id, sessionId);
          this.sessionExpiries.set(client.id, expiresAtMs);
          this.scheduleSessionExpiry(client, expiresAtMs);
          if (!this.hasActiveSession(client, userId)) return false;

          await client.join(`user:${userId}`);
          if (this.isSocketDisconnected(client)) {
            this.clearSession(client.id);
            return false;
          }

          return true;
        }
      );

      if (!admission.authorized) {
        this.revokeSocket(client);
        return;
      }

      if (!admission.result || this.isSocketDisconnected(client)) {
        this.clearSession(client.id);
        return;
      }

      client.emit('session:ready', { socketId: client.id });
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

    const response = await this.withAuthoritativeSession(client, userId, async (tx) => {
      try {
        const result = await this.chatSecurity.persistMessageInTransaction(tx, {
          userId,
          ...payload,
        });

        const message = {
          id: result.message.id,
          conversationId: result.message.conversationId,
          senderId: result.message.senderId,
          text: result.message.text,
          mediaUrls: result.message.mediaUrls,
          timestamp: result.message.createdAt,
        };

        return { success: true, duplicate: result.duplicate, message };
      } catch (error) {
        this.logger.warn(`Rejected chat message: ${this.errorName(error)}`);
        return { success: false, error: 'message_rejected' };
      }
    });

    if (response?.success && !response.duplicate) {
      try {
        await this.chatSecurity.withAuthorizedRealtimeRecipients(
          payload.conversationId,
          async (authorizedUserIds) => {
            await this.emitToActiveSessions(
              authorizedUserIds,
              'message:received',
              response.message
            );
          }
        );
      } catch (error) {
        this.logger.warn(`Chat realtime delivery failed: ${this.errorName(error)}`);
      }

      void this.nudgesService
        .checkChatActivityNudges(payload.conversationId, userId)
        .catch((error) => {
          this.logger.warn(`Chat nudge check failed: ${this.errorName(error)}`);
        });
    }

    return response ?? { success: false, error: 'unauthorized' };
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

    const response = await this.withAuthoritativeSession(client, userId, async (tx) => {
      try {
        await this.chatSecurity.assertConversationAccessInTransaction(
          tx,
          userId,
          payload.conversationId
        );
        await client.join(`conversation:${payload.conversationId}`);
        return { success: true };
      } catch {
        return { success: false, error: 'conversation_unavailable' };
      }
    });

    return response ?? { success: false, error: 'unauthorized' };
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

    const response = await this.withAuthoritativeSession(client, userId, async (tx) => {
      try {
        await this.chatSecurity.assertConversationAccessInTransaction(
          tx,
          userId,
          payload.conversationId
        );
        await client.leave(`conversation:${payload.conversationId}`);
        return { success: true };
      } catch {
        return { success: false, error: 'conversation_unavailable' };
      }
    });

    return response ?? { success: false, error: 'unauthorized' };
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

    const response = await this.withAuthoritativeSession(client, userId, async (tx) => {
      try {
        await this.chatSecurity.withAuthorizedRealtimeRecipientsInTransaction(
          tx,
          payload.conversationId,
          async (authorizedUserIds) => {
            const candidates = this.sessionCandidates(authorizedUserIds);
            await this.sessionAuthority.withActiveSessionsInTransaction(
              tx,
              candidates.map((candidate) => candidate.sessionId),
              (activeSessionIds) => {
                const inactiveSocketIds = candidates
                  .filter((candidate) => !activeSessionIds.has(candidate.sessionId))
                  .map((candidate) => candidate.socketId);
                const authorizedRooms = this.server
                  .to(this.userRooms(authorizedUserIds))
                  .except(client.id);
                if (inactiveSocketIds.length > 0) {
                  authorizedRooms.except(inactiveSocketIds).emit(event, { userId });
                } else {
                  authorizedRooms.emit(event, { userId });
                }
              }
            );
          },
          userId
        );
        return { success: true };
      } catch {
        return { success: false, error: 'conversation_unavailable' };
      }
    });

    return response ?? { success: false, error: 'unauthorized' };
  }

  private async withAuthoritativeSession<T>(
    client: Socket,
    userId: string,
    work: (tx: Prisma.TransactionClient) => Promise<T>
  ) {
    const sessionId = this.connectedSessions.get(client.id);
    if (!sessionId) return null;

    const authority = await this.sessionAuthority.withActiveSession(sessionId, userId, work);
    if (!authority.authorized) {
      this.revokeSocket(client);
      return null;
    }
    return authority.result;
  }

  private async emitToActiveSessions(authorizedUserIds: string[], event: string, payload: unknown) {
    const candidates = this.sessionCandidates(authorizedUserIds);
    await this.sessionAuthority.withActiveSessions(
      candidates.map((candidate) => candidate.sessionId),
      (activeSessionIds) => {
        const inactiveSocketIds = candidates
          .filter((candidate) => !activeSessionIds.has(candidate.sessionId))
          .map((candidate) => candidate.socketId);
        const authorizedRooms = this.server.to(this.userRooms(authorizedUserIds));
        if (inactiveSocketIds.length > 0) {
          authorizedRooms.except(inactiveSocketIds).emit(event, payload);
        } else {
          authorizedRooms.emit(event, payload);
        }
      }
    );
  }

  private sessionCandidates(userIds: string[]) {
    const authorizedUsers = new Set(userIds);
    const candidates: Array<{ socketId: string; sessionId: string }> = [];

    for (const [socketId, userId] of this.connectedUsers.entries()) {
      if (!authorizedUsers.has(userId)) continue;
      const sessionId = this.connectedSessions.get(socketId);
      if (sessionId) candidates.push({ socketId, sessionId });
    }

    return candidates;
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

  private revokeSocket(client: Socket) {
    this.clearSession(client.id);
    client.emit('session:revoked', { reason: 'session_revoked' });
    client.disconnect();
  }

  private clearSession(socketId: string) {
    this.connectedUsers.delete(socketId);
    this.connectedSessions.delete(socketId);
    this.sessionExpiries.delete(socketId);
    this.clearSessionTimer(socketId);
  }

  private clearSessionTimer(socketId: string) {
    const timer = this.sessionExpiryTimers.get(socketId);
    if (!timer) return;

    clearTimeout(timer);
    this.sessionExpiryTimers.delete(socketId);
  }

  private isSocketDisconnected(client: Socket) {
    return client.connected === false;
  }

  private stringClaim(value: unknown) {
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
