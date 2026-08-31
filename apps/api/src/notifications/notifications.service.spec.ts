import { Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as webPush from 'web-push';
import { PrismaService } from '../prisma/prisma.service';
import { NotificationsService } from './notifications.service';

jest.mock('web-push', () => ({
  setVapidDetails: jest.fn(),
  sendNotification: jest.fn(),
}));

const userId = 'user-private-123';
const endpoint = 'https://push.example.com/private-subscription-endpoint';
const title = 'Private achievement title';
const body = 'Private notification body';

function storedSubscription() {
  return {
    id: 'push-token-row-1',
    data: {
      endpoint,
      expirationTime: null,
      keys: {
        p256dh: 'private-p256dh-key',
        auth: 'private-auth-key',
      },
    },
  };
}

function prisma(row: ReturnType<typeof storedSubscription> | null = storedSubscription()) {
  return {
    integrationToken: {
      findFirst: jest.fn().mockResolvedValue(row),
      delete: jest.fn().mockResolvedValue(row),
      upsert: jest.fn().mockResolvedValue(row),
    },
  };
}

function service(
  options: {
    configured?: boolean;
    prisma?: ReturnType<typeof prisma>;
  } = {}
) {
  const configured = options.configured ?? true;
  const values: Record<string, string | undefined> = {
    VAPID_PUBLIC_KEY: configured ? 'test-public-vapid-key' : undefined,
    VAPID_PRIVATE_KEY: configured ? 'test-private-vapid-key' : undefined,
  };
  const config = {
    get: jest.fn((key: string) => values[key]),
  };
  const database = options.prisma ?? prisma();
  return {
    database,
    notifications: new NotificationsService(
      database as unknown as PrismaService,
      config as unknown as ConfigService
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

describe('NotificationsService Web Push privacy boundary', () => {
  const sendNotification = webPush.sendNotification as jest.Mock;
  const setVapidDetails = webPush.setVapidDetails as jest.Mock;

  beforeEach(() => {
    sendNotification.mockReset();
    setVapidDetails.mockReset();
  });

  afterEach(() => {
    jest.restoreAllMocks();
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
    ]) {
      expect(logs).not.toContain(privateValue);
    }
    expect(spies.log).toHaveBeenCalledWith('Push notification delivered');
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
    'removes stale subscriptions on provider status %s without identifier leakage',
    async (statusCode) => {
      const privateMarker = `PRIVATE_EXPIRED_PUSH_${statusCode}`;
      const spies = loggerSpies();
      sendNotification.mockRejectedValue(
        Object.assign(new Error(privateMarker), { statusCode, stack: privateMarker })
      );
      const database = prisma();
      const { notifications } = service({ prisma: database });

      const result = await notifications.sendPushNotification({ userId, title, body });

      expect(result).toEqual({ success: false, reason: 'subscription_expired' });
      expect(database.integrationToken.delete).toHaveBeenCalledWith({
        where: { id: 'push-token-row-1' },
      });
      expect(spies.warn).toHaveBeenCalledWith(
        `Expired push subscription removed status=${statusCode}`
      );
      const logs = serializedLogCalls(spies);
      expect(logs).not.toContain(privateMarker);
      expect(logs).not.toContain(userId);
      expect(logs).not.toContain(endpoint);
    }
  );

  it('returns a truthful disabled state before database/provider access when VAPID is unconfigured', async () => {
    const spies = loggerSpies();
    const database = prisma();
    const { notifications } = service({ configured: false, prisma: database });

    const result = await notifications.sendPushNotification({ userId, title, body });

    expect(result).toEqual({ success: false, reason: 'push_not_configured' });
    expect(database.integrationToken.findFirst).not.toHaveBeenCalled();
    expect(sendNotification).not.toHaveBeenCalled();
    expect(spies.debug).toHaveBeenCalledWith('Push delivery skipped reason=not_configured');
    const logs = serializedLogCalls(spies);
    expect(logs).not.toContain(userId);
    expect(logs).not.toContain(title);
    expect(logs).not.toContain(body);
  });
});
