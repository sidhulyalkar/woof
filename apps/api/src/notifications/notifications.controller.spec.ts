import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { PushSubscriptionDto } from './dto/push-subscription.dto';
import { NotificationsController } from './notifications.controller';
import { NotificationsService } from './notifications.service';

const subscription: PushSubscriptionDto = {
  endpoint: 'https://push.example.com/session-owned-endpoint',
  expirationTime: null,
  keys: {
    p256dh: 'p256dh',
    auth: 'auth',
  },
};
const subscriptionFingerprint = 'a'.repeat(43);

function request(userId = 'session-owner') {
  return { user: { sub: userId } } as AuthenticatedRequest;
}

function service() {
  return {
    getPushSubscriptionStatus: jest.fn().mockResolvedValue({ subscribed: false }),
    subscribePushNotification: jest.fn().mockResolvedValue({ success: true }),
    removeCurrentPushSubscription: jest.fn().mockResolvedValue({ success: true, removed: true }),
    unsubscribePushNotification: jest.fn().mockResolvedValue({ success: true }),
  };
}

describe('NotificationsController push authority', () => {
  it('derives subscription status ownership from the authenticated session', async () => {
    const notifications = service();
    const controller = new NotificationsController(
      notifications as unknown as NotificationsService
    );

    await controller.subscriptionStatus(request('authoritative-user'));

    expect(notifications.getPushSubscriptionStatus).toHaveBeenCalledWith('authoritative-user');
  });

  it('derives subscription ownership from the authenticated session', async () => {
    const notifications = service();
    const controller = new NotificationsController(
      notifications as unknown as NotificationsService
    );

    await controller.subscribe({ subscription }, request('authoritative-user'));

    expect(notifications.subscribePushNotification).toHaveBeenCalledWith(
      'authoritative-user',
      subscription
    );
  });

  it('binds current-browser conditional revocation to the authenticated session', async () => {
    const notifications = service();
    const controller = new NotificationsController(
      notifications as unknown as NotificationsService
    );

    await controller.removeCurrent({ subscriptionFingerprint }, request('authoritative-user'));

    expect(notifications.removeCurrentPushSubscription).toHaveBeenCalledWith(
      'authoritative-user',
      subscriptionFingerprint
    );
  });

  it('removes the authenticated account subscription without requiring credential decryption', async () => {
    const notifications = service();
    const controller = new NotificationsController(
      notifications as unknown as NotificationsService
    );

    await controller.unsubscribe(request('authoritative-user'));

    expect(notifications.unsubscribePushNotification).toHaveBeenCalledWith('authoritative-user');
  });

  it('does not expose the retired arbitrary-target send method', () => {
    const controller = new NotificationsController(service() as unknown as NotificationsService);

    expect('sendPush' in controller).toBe(false);
  });
});
