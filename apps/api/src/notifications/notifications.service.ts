import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Prisma } from '@woof/database';
import * as webPush from 'web-push';
import { PrismaService } from '../prisma/prisma.service';
import { PushSubscriptionDto, SendPushDto } from './dto/push-subscription.dto';

type StoredPushSubscription = {
  endpoint: string;
  expirationTime?: number | null;
  keys: {
    p256dh: string;
    auth: string;
  };
};

type PushDeliveryError = {
  statusCode?: number;
};

function toStoredSubscription(subscription: PushSubscriptionDto): Prisma.InputJsonObject {
  return {
    endpoint: subscription.endpoint,
    expirationTime: subscription.expirationTime ?? null,
    keys: {
      p256dh: subscription.keys.p256dh,
      auth: subscription.keys.auth,
    },
  };
}

function readStoredSubscription(value: Prisma.JsonValue): StoredPushSubscription | null {
  if (!value || Array.isArray(value) || typeof value !== 'object') return null;
  const endpoint = value.endpoint;
  const keys = value.keys;
  if (
    typeof endpoint !== 'string' ||
    !keys ||
    Array.isArray(keys) ||
    typeof keys !== 'object' ||
    typeof keys.p256dh !== 'string' ||
    typeof keys.auth !== 'string'
  ) {
    return null;
  }
  return {
    endpoint,
    expirationTime: typeof value.expirationTime === 'number' ? value.expirationTime : null,
    keys: { p256dh: keys.p256dh, auth: keys.auth },
  };
}

function readPushError(error: unknown): PushDeliveryError {
  if (!error || typeof error !== 'object') return {};
  const candidate = error as Record<string, unknown>;
  return {
    statusCode: typeof candidate.statusCode === 'number' ? candidate.statusCode : undefined,
  };
}

@Injectable()
export class NotificationsService {
  private readonly logger = new Logger(NotificationsService.name);
  private readonly pushConfigured: boolean;

  constructor(
    private prisma: PrismaService,
    private configService: ConfigService
  ) {
    const publicKey = this.configService.get<string>('VAPID_PUBLIC_KEY');
    const privateKey = this.configService.get<string>('VAPID_PRIVATE_KEY');

    this.pushConfigured = Boolean(publicKey && privateKey);

    if (this.pushConfigured) {
      webPush.setVapidDetails('mailto:support@woof.app', publicKey!, privateKey!);
      this.logger.log('Web Push configured');
    } else {
      this.logger.warn('VAPID keys not configured; push delivery is disabled');
    }
  }

  async subscribePushNotification(userId: string, subscription: PushSubscriptionDto) {
    if (!this.pushConfigured) {
      return { success: false, reason: 'push_not_configured' };
    }

    const subscriptionData = toStoredSubscription(subscription);
    const token = await this.prisma.integrationToken.upsert({
      where: {
        userId_provider: {
          userId,
          provider: 'push_subscription',
        },
      },
      create: {
        userId,
        provider: 'push_subscription',
        data: subscriptionData,
        scopes: ['notifications'],
        expiresAt: subscription.expirationTime ? new Date(subscription.expirationTime) : null,
      },
      update: {
        data: subscriptionData,
        expiresAt: subscription.expirationTime ? new Date(subscription.expirationTime) : null,
      },
    });

    this.logger.log('Push subscription saved');
    return token;
  }

  async unsubscribePushNotification(userId: string, endpoint?: string) {
    const subscription = await this.prisma.integrationToken.findFirst({
      where: {
        userId,
        provider: 'push_subscription',
      },
    });

    if (!subscription) {
      return { success: true };
    }

    const subscriptionData = readStoredSubscription(subscription.data);
    if (!endpoint || subscriptionData?.endpoint === endpoint) {
      await this.prisma.integrationToken.delete({
        where: { id: subscription.id },
      });
      this.logger.log('Push subscription removed');
    }

    return { success: true };
  }

  async sendPushNotification(data: SendPushDto) {
    const { userId, title, body, icon, url, data: payload } = data;

    if (!this.pushConfigured) {
      this.logger.debug('Push delivery skipped reason=not_configured');
      return { success: false, reason: 'push_not_configured' };
    }

    const subscription = await this.prisma.integrationToken.findFirst({
      where: {
        userId,
        provider: 'push_subscription',
      },
    });

    if (!subscription) {
      this.logger.debug('Push delivery skipped reason=no_subscription');
      return { success: false, reason: 'no_subscription' };
    }

    const pushSubscription = readStoredSubscription(subscription.data);
    if (!pushSubscription) {
      this.logger.warn('Invalid stored push subscription removed');
      await this.unsubscribePushNotification(userId);
      return { success: false, reason: 'invalid_subscription' };
    }

    const notificationPayload = JSON.stringify({
      title,
      body,
      icon: icon || '/icon-192.png',
      badge: '/badge-72.png',
      data: {
        url: url || '/',
        ...payload,
      },
    });

    try {
      await webPush.sendNotification(pushSubscription, notificationPayload);
      this.logger.log('Push notification delivered');
      return { success: true };
    } catch (error: unknown) {
      const pushError = readPushError(error);
      if (pushError.statusCode === 410 || pushError.statusCode === 404) {
        this.logger.warn(`Expired push subscription removed status=${pushError.statusCode}`);
        await this.unsubscribePushNotification(userId);
        return { success: false, reason: 'subscription_expired' };
      }

      const statusSuffix =
        pushError.statusCode === undefined ? '' : ` status=${pushError.statusCode}`;
      this.logger.error(`Push delivery failed${statusSuffix}`);
      return { success: false, reason: 'delivery_failed' };
    }
  }

  async sendBulkPushNotifications(userIds: string[], data: Omit<SendPushDto, 'userId'>) {
    const results = await Promise.all(
      userIds.map((userId) => this.sendPushNotification({ ...data, userId }))
    );

    const successful = results.filter((result) => result.success).length;
    const failed = results.length - successful;

    this.logger.log(`Bulk push attempted: ${successful} successful, ${failed} not delivered`);
    return { successful, failed, total: userIds.length };
  }

  async sendNudgeNotification(
    userId: string,
    nudgeType: string,
    message: string,
    data?: Record<string, unknown>
  ) {
    return this.sendPushNotification({
      userId,
      title: `New ${nudgeType} suggestion!`,
      body: message,
      icon: '/icon-192.png',
      url: '/notifications',
      data: {
        type: 'nudge',
        nudgeType,
        ...data,
      },
    });
  }

  async sendAchievementNotification(userId: string, title: string, message: string) {
    return this.sendPushNotification({
      userId,
      title: `🏆 ${title}`,
      body: message,
      icon: '/icon-192.png',
      url: '/profile',
      data: { type: 'achievement' },
    });
  }

  async sendEventReminder(userId: string, eventTitle: string, eventId: string) {
    return this.sendPushNotification({
      userId,
      title: 'Event Reminder',
      body: `${eventTitle} is starting soon!`,
      icon: '/icon-192.png',
      url: `/events/${eventId}`,
      data: {
        type: 'event_reminder',
        eventId,
      },
    });
  }
}
