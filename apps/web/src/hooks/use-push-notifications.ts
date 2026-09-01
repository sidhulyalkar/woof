'use client';

import { useState, useEffect, useCallback } from 'react';
import { toast } from 'sonner';
import { notificationsApi } from '@/lib/api';

const VAPID_PUBLIC_KEY = process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY || '';

function urlBase64ToArrayBuffer(base64String: string): ArrayBuffer {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const rawData = window.atob(base64);
  const output = new Uint8Array(rawData.length);

  for (let index = 0; index < rawData.length; index += 1) {
    output[index] = rawData.charCodeAt(index);
  }

  return output.buffer;
}

function arrayBufferToBase64Url(buffer: ArrayBuffer) {
  let binary = '';
  for (const byte of new Uint8Array(buffer)) {
    binary += String.fromCharCode(byte);
  }
  return window.btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function canonicalPushSubscription(subscription: PushSubscription) {
  const serialized = subscription.toJSON();
  const p256dh = serialized.keys?.p256dh;
  const auth = serialized.keys?.auth;
  if (!p256dh || !auth) {
    throw new Error('Push subscription key material is unavailable');
  }

  return JSON.stringify({
    endpoint: subscription.endpoint,
    expirationTime: subscription.expirationTime ?? null,
    keys: { p256dh, auth },
  });
}

async function subscriptionFingerprint(subscription: PushSubscription) {
  const digest = await window.crypto.subtle.digest(
    'SHA-256',
    new TextEncoder().encode(canonicalPushSubscription(subscription))
  );
  return arrayBufferToBase64Url(digest);
}

export function usePushNotifications() {
  const [isSupported, setIsSupported] = useState(false);
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const supported =
      typeof window !== 'undefined' &&
      'serviceWorker' in navigator &&
      'PushManager' in window &&
      'Notification' in window &&
      Boolean(window.crypto?.subtle);

    setIsSupported(supported);
    if (supported) {
      setPermission(Notification.permission);

      void navigator.serviceWorker.ready.then(async (registration) => {
        try {
          const browserSubscription = await registration.pushManager.getSubscription();
          const serverStatus = await notificationsApi.status();

          if (
            !browserSubscription ||
            !serverStatus.subscribed ||
            !serverStatus.subscriptionFingerprint
          ) {
            setIsSubscribed(false);
            return;
          }

          const browserFingerprint = await subscriptionFingerprint(browserSubscription);
          setIsSubscribed(browserFingerprint === serverStatus.subscriptionFingerprint);
        } catch {
          setIsSubscribed(false);
          console.error('Push subscription status reconciliation failed');
        }
      });
    }
  }, []);

  const requestPermission = useCallback(async () => {
    if (!isSupported) {
      toast.error('Push notifications are not supported');
      return 'denied' as NotificationPermission;
    }

    try {
      const result = await Notification.requestPermission();
      setPermission(result);
      return result;
    } catch {
      console.error('Push notification permission request failed');
      toast.error('Failed to request notification permission');
      return 'denied' as NotificationPermission;
    }
  }, [isSupported]);

  const subscribe = useCallback(async () => {
    if (!isSupported) {
      toast.error('Push notifications are not supported');
      return;
    }

    let browserSubscription: PushSubscription | null = null;
    let browserFingerprint: string | null = null;
    try {
      setIsLoading(true);

      if (permission !== 'granted') {
        const result = await requestPermission();
        if (result !== 'granted') return;
      }

      if (!VAPID_PUBLIC_KEY) {
        toast.error('Push notifications are not configured');
        return;
      }

      const registration = await navigator.serviceWorker.ready;
      browserSubscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToArrayBuffer(VAPID_PUBLIC_KEY),
      });
      browserFingerprint = await subscriptionFingerprint(browserSubscription);

      const serverResult = await notificationsApi.subscribe(browserSubscription.toJSON());
      if (!serverResult.success) {
        try {
          await browserSubscription.unsubscribe();
        } catch {
          console.error('Push local subscription rollback failed');
        }
        browserSubscription = null;
        browserFingerprint = null;
        setIsSubscribed(false);
        toast.error('Push notifications are not available right now');
        return;
      }

      setIsSubscribed(true);
      toast.success('Push notifications enabled!');
    } catch {
      console.error('Push notification subscription failed');

      // An ambiguous network failure may occur after the server committed. Conditional
      // compare-and-delete removes only this exact attempted browser subscription and cannot
      // erase a replacement whose endpoint or Push keys changed concurrently.
      if (browserFingerprint) {
        try {
          await notificationsApi.unsubscribeCurrent(browserFingerprint);
        } catch {
          console.error('Push current-browser server rollback failed');
        }
      }
      if (browserSubscription) {
        try {
          await browserSubscription.unsubscribe();
        } catch {
          console.error('Push local subscription rollback failed');
        }
      }
      setIsSubscribed(false);
      toast.error('Failed to enable push notifications');
    } finally {
      setIsLoading(false);
    }
  }, [isSupported, permission, requestPermission]);

  const unsubscribe = useCallback(async () => {
    if (!isSupported) return;

    try {
      setIsLoading(true);

      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      if (!subscription) {
        setIsSubscribed(false);
        toast.success('Push notifications disabled');
        return;
      }

      const fingerprint = await subscriptionFingerprint(subscription);
      const serverResult = await notificationsApi.unsubscribeCurrent(fingerprint);
      if (!serverResult.success) {
        toast.error('Failed to disable push notifications');
        return;
      }

      try {
        await subscription.unsubscribe();
      } catch {
        // If the exact matching row was removed, server delivery authority is already revoked.
        // If server state changed first, conditional removal was a no-op.
        console.error('Push local unsubscribe cleanup failed');
      }

      setIsSubscribed(false);
      toast.success('Push notifications disabled');
    } catch {
      console.error('Push notification unsubscribe failed');
      toast.error('Failed to disable push notifications');
    } finally {
      setIsLoading(false);
    }
  }, [isSupported]);

  return {
    isSupported,
    permission,
    isSubscribed,
    isLoading,
    requestPermission,
    subscribe,
    unsubscribe,
  };
}
