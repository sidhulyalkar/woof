import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as webPush from 'web-push';
import { PushSubscriptionDto } from './dto/push-subscription.dto';
import { pushSubscriptionFingerprint, PushSubscriptionStore } from './push-subscription.store';

type PushDeliveryError = {
  statusCode?: number;
};

type PushNotificationInput = {
  userId: string;
  title: string;
  body: string;
  data?: Record<string, unknown>;
  icon?: string;
  url?: string;
};

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
  private readonly vapidConfigured: boolean;
  private readonly encryptionConfigured: boolean;

  constructor(
    private readonly configService: ConfigService,
    private readonly subscriptions: PushSubscriptionStore
  ) {
    const publicKey = this.configService.get<string>('VAPID_PUBLIC_KEY');
    const privateKey = this.configService.get<string>('VAPID_PRIVATE_KEY');

    this.vapidConfigured = Boolean(publicKey && privateKey);
    this.encryptionConfigured = this.subscriptions.encryptionConfigured();

    if (this.vapidConfigured && this.encryptionConfigured) {
      webPush.setVapidDetails('mailto:support@woof.app', publicKey!, privateKey!);
      this.logger.log('Web Push configured with encrypted subscription storage');
    } else if (!this.vapidConfigured) {
      this.logger.warn('VAPID keys not configured; push delivery is disabled');
    } else {
      this.logger.warn('Push subscription encryption not configured; push delivery is disabled');
    }
  }

  async getPushSubscriptionStatus(userId: string) {
    if (!this.vapidConfigured || !this.encryptionConfigured) {
      return { subscribed: false };
    }

    const stored = await this.subscriptions.get(userId);
    if (stored.state === 'INVALID') {
      await this.subscriptions.removeInvalidCurrent(userId);
      return { subscribed: false };
    }
    if (stored.state !== 'USABLE') {
      return { subscribed: false };
    }

    return {
      subscribed: true,
      subscriptionFingerprint: pushSubscriptionFingerprint(stored.subscription),
    };
  }

  async subscribePushNotification(userId: string, subscription: PushSubscriptionDto) {
    if (!this.vapidConfigured) {
      return { success: false, reason: 'push_not_configured' };
    }
    if (!this.encryptionConfigured) {
      return { success: false, reason: 'push_encryption_not_configured' };
    }

    await this.subscriptions.put(userId, subscription);
    this.logger.log('Push subscription saved in encrypted storage');
    return { success: true };
  }

  async removeCurrentPushSubscription(userId: string, subscriptionFingerprint: string) {
    const removed = await this.subscriptions.removeIfFingerprint(userId, subscriptionFingerprint);
    if (removed) {
      this.logger.log('Current browser push subscription removed');
    }
    return { success: true, removed };
  }

  async unsubscribePushNotification(userId: string) {
    await this.subscriptions.remove(userId);
    this.logger.log('Push subscription removed');
    return { success: true };
  }

  async sendPushNotification(data: PushNotificationInput) {
    const { userId, title, body, icon, url, data: payload } = data;

    if (!this.vapidConfigured) {
      this.logger.debug('Push delivery skipped reason=not_configured');
      return { success: false, reason: 'push_not_configured' };
    }
    if (!this.encryptionConfigured) {
      this.logger.debug('Push delivery skipped reason=encryption_not_configured');
      return { success: false, reason: 'push_encryption_not_configured' };
    }

    const stored = await this.subscriptions.get(userId);
    if (stored.state === 'MISSING') {
      this.logger.debug('Push delivery skipped reason=no_subscription');
      return { success: false, reason: 'no_subscription' };
    }
    if (stored.state === 'ENCRYPTION_UNAVAILABLE') {
      this.logger.debug('Push delivery skipped reason=encryption_unavailable');
      return { success: false, reason: 'push_encryption_not_configured' };
    }
    if (stored.state === 'LEGACY_MIGRATION_REQUIRED') {
      this.logger.warn('Legacy push subscription requires operator migration');
      return { success: false, reason: 'legacy_migration_required' };
    }
    if (stored.state === 'CONCURRENT_CHANGE') {
      this.logger.debug('Push delivery skipped reason=subscription_changed');
      return { success: false, reason: 'subscription_changed' };
    }
    if (stored.state === 'INVALID') {
      const removed = await this.subscriptions.removeInvalidCurrent(userId);
      this.logger.warn(
        removed
          ? 'Invalid stored push subscription removed'
          : 'Invalid stored push subscription cleanup skipped'
      );
      return { success: false, reason: 'invalid_subscription' };
    }

    if (stored.migratedLegacy) {
      this.logger.log('Legacy push subscription migrated to encrypted storage');
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
      await webPush.sendNotification(stored.subscription, notificationPayload);
      this.logger.log('Push notification delivered');
      return { success: true };
    } catch (error: unknown) {
      const pushError = readPushError(error);
      if (pushError.statusCode === 410 || pushError.statusCode === 404) {
        const removed = await this.subscriptions.removeIfFingerprint(
          userId,
          pushSubscriptionFingerprint(stored.subscription)
        );
        this.logger.warn(
          `Expired push subscription cleanup status=${pushError.statusCode} removed=${removed ? 'yes' : 'no'}`
        );
        return { success: false, reason: 'subscription_expired' };
      }

      const statusSuffix =
        pushError.statusCode === undefined ? '' : ` status=${pushError.statusCode}`;
      this.logger.error(`Push delivery failed${statusSuffix}`);
      return { success: false, reason: 'delivery_failed' };
    }
  }

  async sendBulkPushNotifications(userIds: string[], data: Omit<PushNotificationInput, 'userId'>) {
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
