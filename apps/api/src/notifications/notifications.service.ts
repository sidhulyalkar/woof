import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as webPush from 'web-push';
import { PrismaService } from '../prisma/prisma.service';
import { PushSubscriptionDto, SendPushDto } from './dto/push-subscription.dto';

@Injectable()
export class NotificationsService {
  private readonly logger = new Logger(NotificationsService.name);

  constructor(
    private prisma: PrismaService,
    private configService: ConfigService,
  ) {
    // Configure web-push with VAPID keys
    const publicKey = this.configService.get('VAPID_PUBLIC_KEY');
    const privateKey = this.configService.get('VAPID_PRIVATE_KEY');

    if (publicKey && privateKey) {
      webPush.setVapidDetails(
        'mailto:support@woof.app',
        publicKey,
        privateKey,
      );
      this.logger.log('Web Push configured successfully');
    } else {
      this.logger.warn('VAPID keys not configured - push notifications disabled');
    }
  }

  /**
   * Subscribe a user to push notifications
   */
  async subscribePushNotification(
    userId: string,
    subscription: PushSubscriptionDto,
  ) {
    try {
      // Store subscription in IntegrationToken table
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
    } catch (error) {
      this.logger.error(
        `Failed to save push subscription: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  /**
   * Unsubscribe from push notifications
   */
  async unsubscribePushNotification(userId: string, endpoint: string) {
    try {
      // Find and delete the subscription
      const subscription = await this.prisma.integrationToken.findFirst({
        where: {
          userId,
          provider: 'push_subscription',
        },
      });

      if (subscription) {
        const subscriptionData = subscription.data as any;
        if (subscriptionData.endpoint === endpoint) {
          await this.prisma.integrationToken.delete({
            where: { id: subscription.id },
          });
          this.logger.log(`Push subscription removed for user ${userId}`);
        }
      }

      return { success: true };
    } catch (error) {
      this.logger.error(
        `Failed to remove push subscription: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  /**
   * Send a push notification to a user
   */
  async sendPushNotification(data: SendPushDto) {
    const { userId, title, body, icon, url, data: payload } = data;

    try {
      // Get user's push subscription
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

      // Prepare notification payload
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

      // Send push notification
      await webPush.sendNotification(pushSubscription, notificationPayload);

      this.logger.log(`Push notification sent to user ${userId}: ${title}`);
      return { success: true };
    } catch (error) {
      // Handle expired subscriptions
      if (error.statusCode === 410) {
        this.logger.warn(`Push subscription expired for user ${userId}, removing...`);
        await this.unsubscribePushNotification(userId, '');
        return { success: false, reason: 'subscription_expired' };
      }

      this.logger.error(
        `Failed to send push notification: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  /**
   * Send push notification to multiple users
   */
  async sendBulkPushNotifications(
    userIds: string[],
    data: Omit<SendPushDto, 'userId'>,
  ) {
    const results = await Promise.allSettled(
      userIds.map((userId) =>
        this.sendPushNotification({ ...data, userId }),
      ),
    );

    const successful = results.filter((r) => r.status === 'fulfilled').length;
    const failed = results.filter((r) => r.status === 'rejected').length;

    this.logger.log(
      `Bulk push sent: ${successful} successful, ${failed} failed`,
    );

    return { successful, failed, total: userIds.length };
  }

  /**
   * Send nudge notification
   */
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

  /**
   * Send achievement notification
   */
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
      data: {
        type: 'achievement',
      },
    });
  }

  /**
   * Send event reminder
   */
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
