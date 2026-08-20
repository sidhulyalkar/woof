import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as webPush from 'web-push';
import { PrismaService } from '../prisma/prisma.service';
import { PushSubscriptionDto, SendPushDto } from './dto/push-subscription.dto';

@Injectable()
export class NotificationsService {
  private readonly logger = new Logger(NotificationsService.name);
  private readonly pushConfigured: boolean;

  constructor(
    private prisma: PrismaService,
    private configService: ConfigService,
  ) {
    const publicKey = this.configService.get<string>('VAPID_PUBLIC_KEY');
    const privateKey = this.configService.get<string>('VAPID_PRIVATE_KEY');

    this.pushConfigured = Boolean(publicKey && privateKey);

    if (this.pushConfigured) {
      webPush.setVapidDetails(
        'mailto:support@woof.app',
        publicKey!,
        privateKey!,
      );
      this.logger.log('Web Push configured');
    } else {
      this.logger.warn('VAPID keys not configured; push delivery is disabled');
    }
  }

  async subscribePushNotification(
    userId: string,
    subscription: PushSubscriptionDto,
  ) {
    if (!this.pushConfigured) {
      return { success: false, reason: 'push_not_configured' };
    }

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
        data: subscription as any,
        scopes: ['notifications'],
        expiresAt: subscription.expirationTime
          ? new Date(subscription.expirationTime)
          : null,
      },
      update: {
        data: subscription as any,
        expiresAt: subscription.expirationTime
          ? new Date(subscription.expirationTime)
          : null,
      },
    });

    this.logger.log(`Push subscription saved for user ${userId}`);
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

    const subscriptionData = subscription.data as any;
    if (!endpoint || subscriptionData.endpoint === endpoint) {
      await this.prisma.integrationToken.delete({
        where: { id: subscription.id },
      });
      this.logger.log(`Push subscription removed for user ${userId}`);
    }

    return { success: true };
  }

  async sendPushNotification(data: SendPushDto) {
    const { userId, title, body, icon, url, data: payload } = data;

    if (!this.pushConfigured) {
      this.logger.debug(`Push skipped for user ${userId}: VAPID not configured`);
      return { success: false, reason: 'push_not_configured' };
    }

    const subscription = await this.prisma.integrationToken.findFirst({
      where: {
        userId,
        provider: 'push_subscription',
      },
    });

    if (!subscription) {
      this.logger.debug(`No push subscription found for user ${userId}`);
      return { success: false, reason: 'no_subscription' };
    }

    const pushSubscription = subscription.data as any;
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
      this.logger.log(`Push notification sent to user ${userId}: ${title}`);
      return { success: true };
    } catch (error: any) {
      if (error?.statusCode === 410 || error?.statusCode === 404) {
        this.logger.warn(`Expired push subscription removed for user ${userId}`);
        await this.unsubscribePushNotification(userId);
        return { success: false, reason: 'subscription_expired' };
      }

      this.logger.error(
        `Failed to send push notification: ${error?.message || 'unknown error'}`,
        error?.stack,
      );
      return { success: false, reason: 'delivery_failed' };
    }
  }

  async sendBulkPushNotifications(
    userIds: string[],
    data: Omit<SendPushDto, 'userId'>,
  ) {
    const results = await Promise.all(
      userIds.map((userId) => this.sendPushNotification({ ...data, userId })),
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
    data?: Record<string, any>,
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

  async sendAchievementNotification(
    userId: string,
    title: string,
    message: string,
  ) {
    return this.sendPushNotification({
      userId,
      title: `🏆 ${title}`,
      body: message,
      icon: '/icon-192.png',
      url: '/profile',
      data: { type: 'achievement' },
    });
  }

  async sendEventReminder(
    userId: string,
    eventTitle: string,
    eventId: string,
  ) {
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
