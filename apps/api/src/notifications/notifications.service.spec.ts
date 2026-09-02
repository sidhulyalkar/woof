import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as webPush from 'web-push';
import {
  pushSubscriptionFingerprint,
  PushSubscriptionReadResult,
  PushSubscriptionStore,
} from './push-subscription.store';
import { NotificationsService } from './notifications.service';

jest.mock('web-push', () => ({
  setVapidDetails: jest.fn(),
  sendNotification: jest.fn(),
}));

const userId = 'user-private-123';
const endpoint = 'https://push.example.com/private-subscription-endpoint';
const title = 'Private achievement title';
const body = 'Private notification body';
const subscription = {
  endpoint,
  expirationTime: null,
  keys: {
    p256dh: 'private-p256dh-key',
    auth: 'private-auth-key',
  },
};
const subscriptionFingerprint = pushSubscriptionFingerprint(subscription);

function pushStore(
  options: {
    encryptionConfigured?: boolean;
    readState?: PushSubscriptionReadResult;
    conditionalRemove?: boolean;
    invalidRemove?: boolean;
  } = {}
) {
  const readState: PushSubscriptionReadResult = options.readState ?? {
    state: 'USABLE',
    subscription,
    migratedLegacy: false,
  };
  return {
    encryptionConfigured: jest.fn(() => options.encryptionConfigured ?? true),
    put: jest.fn().mockResolvedValue(undefined),
    get: jest.fn().mockResolvedValue(readState),
    remove: jest.fn().mockResolvedValue(undefined),
    removeIfFingerprint: jest.fn().mockResolvedValue(options.conditionalRemove ?? true),
    removeInvalidCurrent: jest.fn().mockResolvedValue(options.invalidRemove ?? true),
  };
}

function service(
  options: {
    vapidConfigured?: boolean;
    encryptionConfigured?: boolean;
    readState?: PushSubscriptionReadResult;
    conditionalRemove?: boolean;
    invalidRemove?: boolean;
  } = {}
) {
  const vapidConfigured = options.vapidConfigured ?? true;
  const values: Record<string, string | undefined> = {
    VAPID_PUBLIC_KEY: vapidConfigured ? 'test-public-vapid-key' : undefined,
    VAPID_PRIVATE_KEY: vapidConfigured ? 'test-private-vapid-key' : undefined,
  };
  const config = {
    get: jest.fn((key: string) => values[key]),
  };
  const subscriptions = pushStore({
    encryptionConfigured: options.encryptionConfigured,
    readState: options.readState,
    conditionalRemove: options.conditionalRemove,
    invalidRemove: options.invalidRemove,
  });
  return {
    subscriptions,
    notifications: new NotificationsService(
      config as unknown as ConfigService,
      subscriptions as unknown as PushSubscriptionStore
    ),
  };
}

function loggerSpies() {
  return {
    log: jest.spyOn(Logger.prototype, 'log').mockImplementation(() => undefined),
    debug: jest.spyOn(Logger.prototype, 'debug').mockImplementation(() => undefined),
    warn: jest.spyOn(Logger.prototype, 'warn').mockImplementation(() => undefined),
    error: jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined),
  };
}

function serializedLogCalls(spies: ReturnType<typeof loggerSpies>) {
  return JSON.stringify([
    ...spies.log.mock.calls,
    ...spies.debug.mock.calls,
    ...spies.warn.mock.calls,
    ...spies.error.mock.calls,
  ]);
}

describe('NotificationsService Web Push privacy and encrypted storage boundary', () => {
  const sendNotification = webPush.sendNotification as jest.Mock;
  const setVapidDetails = webPush.setVapidDetails as jest.Mock;

  beforeEach(() => {
    sendNotification.mockReset();
    setVapidDetails.mockReset();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('reports subscription status with a full-material fingerprint, not private Push material', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service();

    await expect(notifications.getPushSubscriptionStatus(userId)).resolves.toEqual({
      subscribed: true,
      subscriptionFingerprint,
    });
    expect(subscriptions.get).toHaveBeenCalledWith(userId);
    expect(subscriptionFingerprint).not.toContain(endpoint);
    expect(subscriptionFingerprint).not.toContain(subscription.keys.p256dh);
    expect(subscriptionFingerprint).not.toContain(subscription.keys.auth);
    expect(serializedLogCalls(spies)).not.toContain(subscriptionFingerprint);
  });

  it('cleans invalid server state with exact-row authority while reporting unsubscribed', async () => {
    const { notifications, subscriptions } = service({ readState: { state: 'INVALID' } });

    await expect(notifications.getPushSubscriptionStatus(userId)).resolves.toEqual({
      subscribed: false,
    });
    expect(subscriptions.removeInvalidCurrent).toHaveBeenCalledWith(userId);
    expect(subscriptions.remove).not.toHaveBeenCalled();
  });

  it('reports legacy migration required as unsubscribed without deleting plaintext state', async () => {
    const { notifications, subscriptions } = service({
      readState: { state: 'LEGACY_MIGRATION_REQUIRED' },
    });

    await expect(notifications.getPushSubscriptionStatus(userId)).resolves.toEqual({
      subscribed: false,
    });
    expect(subscriptions.removeInvalidCurrent).not.toHaveBeenCalled();
    expect(subscriptions.remove).not.toHaveBeenCalled();
  });

  it('reports unsubscribed without reading credentials when Push configuration is unavailable', async () => {
    const { notifications, subscriptions } = service({ encryptionConfigured: false });

    await expect(notifications.getPushSubscriptionStatus(userId)).resolves.toEqual({
      subscribed: false,
    });
    expect(subscriptions.get).not.toHaveBeenCalled();
  });

  it('stores subscriptions through the encrypted store and exposes no credential row', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service();

    const result = await notifications.subscribePushNotification(userId, subscription);

    expect(result).toEqual({ success: true });
    expect(subscriptions.put).toHaveBeenCalledWith(userId, subscription);
    expect(spies.log).toHaveBeenCalledWith('Push subscription saved in encrypted storage');
    expect(serializedLogCalls(spies)).not.toContain(endpoint);
  });

  it('uses fingerprint-bound removal for current-browser revocation', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service();

    await expect(
      notifications.removeCurrentPushSubscription(userId, subscriptionFingerprint)
    ).resolves.toEqual({ success: true, removed: true });
    expect(subscriptions.removeIfFingerprint).toHaveBeenCalledWith(userId, subscriptionFingerprint);
    expect(spies.log).toHaveBeenCalledWith('Current browser push subscription removed');
    expect(serializedLogCalls(spies)).not.toContain(subscriptionFingerprint);
  });

  it('treats a fingerprint mismatch as a safe no-op instead of account-wide deletion', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({ conditionalRemove: false });

    await expect(
      notifications.removeCurrentPushSubscription(userId, subscriptionFingerprint)
    ).resolves.toEqual({ success: true, removed: false });
    expect(subscriptions.remove).not.toHaveBeenCalled();
    expect(spies.log).not.toHaveBeenCalledWith('Current browser push subscription removed');
  });

  it('configures VAPID delivery and sends the intended payload without logging private content', async () => {
    const spies = loggerSpies();
    sendNotification.mockResolvedValue({ statusCode: 201 });
    const { notifications } = service();

    const result = await notifications.sendPushNotification({
      userId,
      title,
      body,
      url: '/private-destination',
      data: { privateContext: 'only-for-device-payload' },
    });

    expect(result).toEqual({ success: true });
    expect(setVapidDetails).toHaveBeenCalledWith(
      'mailto:support@woof.app',
      'test-public-vapid-key',
      'test-private-vapid-key'
    );
    expect(sendNotification).toHaveBeenCalledTimes(1);
    const sentPayload = String(sendNotification.mock.calls[0]?.[1]);
    expect(sentPayload).toContain(title);
    expect(sentPayload).toContain(body);

    const logs = serializedLogCalls(spies);
    for (const privateValue of [
      userId,
      endpoint,
      title,
      body,
      'private-p256dh-key',
      'private-auth-key',
      'only-for-device-payload',
      subscriptionFingerprint,
    ]) {
      expect(logs).not.toContain(privateValue);
    }
    expect(spies.log).toHaveBeenCalledWith('Push notification delivered');
  });

  it('fails delivery closed when legacy plaintext compatibility has ended', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({
      readState: { state: 'LEGACY_MIGRATION_REQUIRED' },
    });

    await expect(notifications.sendPushNotification({ userId, title, body })).resolves.toEqual({
      success: false,
      reason: 'legacy_migration_required',
    });
    expect(sendNotification).not.toHaveBeenCalled();
    expect(subscriptions.remove).not.toHaveBeenCalled();
    expect(subscriptions.removeInvalidCurrent).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith('Legacy push subscription requires operator migration');
  });

  it('reduces arbitrary provider failures to status-only telemetry', async () => {
    const privateMarker = 'PRIVATE_PUSH_PROVIDER_MESSAGE endpoint-and-request-details';
    const spies = loggerSpies();
    sendNotification.mockRejectedValue(
      Object.assign(new Error(privateMarker), {
        statusCode: 500,
        stack: `stack:${privateMarker}`,
      })
    );
    const { notifications } = service();

    const result = await notifications.sendPushNotification({ userId, title, body });

    expect(result).toEqual({ success: false, reason: 'delivery_failed' });
    expect(spies.error).toHaveBeenCalledWith('Push delivery failed status=500');
    const logs = serializedLogCalls(spies);
    expect(logs).not.toContain(privateMarker);
    expect(logs).not.toContain(userId);
    expect(logs).not.toContain(title);
    expect(logs).not.toContain(body);
    expect(logs).not.toContain(endpoint);
  });

  it.each([404, 410])(
    'removes only the exact expired subscription on provider status %s without identifier leakage',
    async (statusCode) => {
      const privateMarker = `PRIVATE_EXPIRED_PUSH_${statusCode}`;
      const spies = loggerSpies();
      sendNotification.mockRejectedValue(
        Object.assign(new Error(privateMarker), { statusCode, stack: privateMarker })
      );
      const { notifications, subscriptions } = service();

      const result = await notifications.sendPushNotification({ userId, title, body });

      expect(result).toEqual({ success: false, reason: 'subscription_expired' });
      expect(subscriptions.removeIfFingerprint).toHaveBeenCalledWith(
        userId,
        subscriptionFingerprint
      );
      expect(subscriptions.remove).not.toHaveBeenCalled();
      expect(spies.warn).toHaveBeenCalledWith(
        `Expired push subscription cleanup status=${statusCode} removed=yes`
      );
      const logs = serializedLogCalls(spies);
      expect(logs).not.toContain(privateMarker);
      expect(logs).not.toContain(userId);
      expect(logs).not.toContain(endpoint);
      expect(logs).not.toContain(subscriptionFingerprint);
    }
  );

  it('does not erase a replacement when provider-expiry cleanup loses the conditional race', async () => {
    const spies = loggerSpies();
    sendNotification.mockRejectedValue(Object.assign(new Error('expired'), { statusCode: 410 }));
    const { notifications, subscriptions } = service({ conditionalRemove: false });

    await expect(notifications.sendPushNotification({ userId, title, body })).resolves.toEqual({
      success: false,
      reason: 'subscription_expired',
    });
    expect(subscriptions.remove).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith(
      'Expired push subscription cleanup status=410 removed=no'
    );
  });

  it('removes invalid encrypted rows only through exact invalid-row cleanup', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({ readState: { state: 'INVALID' } });

    const result = await notifications.sendPushNotification({ userId, title, body });

    expect(result).toEqual({ success: false, reason: 'invalid_subscription' });
    expect(subscriptions.removeInvalidCurrent).toHaveBeenCalledWith(userId);
    expect(subscriptions.remove).not.toHaveBeenCalled();
    expect(sendNotification).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith('Invalid stored push subscription removed');
  });

  it('does not claim invalid-row removal when a concurrent replacement wins', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({
      readState: { state: 'INVALID' },
      invalidRemove: false,
    });

    await expect(notifications.sendPushNotification({ userId, title, body })).resolves.toEqual({
      success: false,
      reason: 'invalid_subscription',
    });
    expect(subscriptions.remove).not.toHaveBeenCalled();
    expect(spies.warn).toHaveBeenCalledWith('Invalid stored push subscription cleanup skipped');
  });

  it('records legacy migration without credential or owner leakage', async () => {
    const spies = loggerSpies();
    sendNotification.mockResolvedValue({ statusCode: 201 });
    const { notifications } = service({
      readState: { state: 'USABLE', subscription, migratedLegacy: true },
    });

    await notifications.sendPushNotification({ userId, title, body });

    expect(spies.log).toHaveBeenCalledWith(
      'Legacy push subscription migrated to encrypted storage'
    );
    const logs = serializedLogCalls(spies);
    expect(logs).not.toContain(userId);
    expect(logs).not.toContain(endpoint);
    expect(logs).not.toContain(subscription.keys.p256dh);
    expect(logs).not.toContain(subscription.keys.auth);
  });

  it('returns a truthful disabled state before storage/provider access when VAPID is unconfigured', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({ vapidConfigured: false });

    const result = await notifications.sendPushNotification({ userId, title, body });

    expect(result).toEqual({ success: false, reason: 'push_not_configured' });
    expect(subscriptions.get).not.toHaveBeenCalled();
    expect(sendNotification).not.toHaveBeenCalled();
    expect(spies.debug).toHaveBeenCalledWith('Push delivery skipped reason=not_configured');
  });

  it('fails closed before storage/provider access when subscription encryption is unconfigured', async () => {
    const spies = loggerSpies();
    const { notifications, subscriptions } = service({ encryptionConfigured: false });

    const result = await notifications.sendPushNotification({ userId, title, body });
    const subscribeResult = await notifications.subscribePushNotification(userId, subscription);

    expect(result).toEqual({ success: false, reason: 'push_encryption_not_configured' });
    expect(subscribeResult).toEqual({
      success: false,
      reason: 'push_encryption_not_configured',
    });
    expect(subscriptions.get).not.toHaveBeenCalled();
    expect(subscriptions.put).not.toHaveBeenCalled();
    expect(sendNotification).not.toHaveBeenCalled();
    expect(setVapidDetails).not.toHaveBeenCalled();
    expect(spies.debug).toHaveBeenCalledWith(
      'Push delivery skipped reason=encryption_not_configured'
    );
  });

  it('keeps account-wide unsubscribe available when VAPID and encryption are unavailable', async () => {
    const { notifications, subscriptions } = service({
      vapidConfigured: false,
      encryptionConfigured: false,
    });

    await expect(notifications.unsubscribePushNotification(userId)).resolves.toEqual({
      success: true,
    });
    expect(subscriptions.remove).toHaveBeenCalledWith(userId);
  });
});
