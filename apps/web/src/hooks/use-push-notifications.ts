"use client"

import { useState, useEffect, useCallback } from "react"
import { notificationsApi } from "@/lib/api"
import { toast } from "sonner"

const VAPID_PUBLIC_KEY =
  process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY ||
  "BLhIwnkmfI6BbgwwJp_08yLyUANn0Ja6C_MWJ1c5jyCefZ_8m4wxbXBgxhF1esCkc-3cCVuyZwyAZ7v5ccSVAWw"

interface PushSubscription {
  endpoint: string
  expirationTime?: number | null
  keys: {
    p256dh: string
    auth: string
  }
}

interface UsePushNotificationsReturn {
  isSupported: boolean
  isSubscribed: boolean
  isLoading: boolean
  permission: NotificationPermission
  subscribe: () => Promise<void>
  unsubscribe: () => Promise<void>
  requestPermission: () => Promise<NotificationPermission>
}

function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/")

  const rawData = window.atob(base64)
  const outputArray = new Uint8Array(rawData.length)

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i)
  }
  return outputArray
}

export function usePushNotifications(): UsePushNotificationsReturn {
  const [isSupported, setIsSupported] = useState(false)
  const [isSubscribed, setIsSubscribed] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [permission, setPermission] = useState<NotificationPermission>("default")

  // Check if push notifications are supported
  useEffect(() => {
    const checkSupport = async () => {
      const supported =
        "serviceWorker" in navigator &&
        "PushManager" in window &&
        "Notification" in window

      setIsSupported(supported)

      if (supported) {
        setPermission(Notification.permission)
        await checkSubscription()
      }

      setIsLoading(false)
    }

    checkSupport()
  }, [])

  // Check current subscription status
  const checkSubscription = useCallback(async () => {
    try {
      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.getSubscription()
      setIsSubscribed(!!subscription)
      return !!subscription
    } catch (error) {
      console.error("Error checking subscription:", error)
      return false
    }
  }, [])

  // Request notification permission
  const requestPermission = useCallback(async (): Promise<NotificationPermission> => {
    if (!isSupported) {
      toast.error("Push notifications are not supported in this browser")
      return "denied"
    }

    try {
      const result = await Notification.requestPermission()
      setPermission(result)

      if (result === "granted") {
        toast.success("Notification permission granted!")
      } else if (result === "denied") {
        toast.error("Notification permission denied")
      }

      return result
    } catch (error) {
      console.error("Error requesting permission:", error)
      toast.error("Failed to request notification permission")
      return "denied"
    }
  }, [isSupported])

  // Subscribe to push notifications
  const subscribe = useCallback(async () => {
    if (!isSupported) {
      toast.error("Push notifications are not supported")
      return
    }

    try {
      setIsLoading(true)

      // Request permission if not granted
      if (permission !== "granted") {
        const result = await requestPermission()
        if (result !== "granted") {
          setIsLoading(false)
          return
        }
      }

      // Get service worker registration
      const registration = await navigator.serviceWorker.ready

      // Subscribe to push manager
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY),
      })

      // Send subscription to backend
      const subscriptionJson = subscription.toJSON() as PushSubscription

      await notificationsApi.subscribe(subscriptionJson)

      setIsSubscribed(true)
      toast.success("Push notifications enabled!")
    } catch (error: any) {
      console.error("Error subscribing to push notifications:", error)

      if (error?.message?.includes("Registration failed")) {
        toast.error("Service worker registration failed. Please refresh the page.")
      } else {
        toast.error("Failed to enable push notifications")
      }
    } finally {
      setIsLoading(false)
    }
  }, [isSupported, permission, requestPermission])

  // Unsubscribe from push notifications
  const unsubscribe = useCallback(async () => {
    if (!isSupported) {
      return
    }

    try {
      setIsLoading(true)

      const registration = await navigator.serviceWorker.ready
      const subscription = await registration.pushManager.getSubscription()

      if (subscription) {
        await subscription.unsubscribe()

        // Notify backend
        await notificationsApi.unsubscribe()

        setIsSubscribed(false)
        toast.success("Push notifications disabled")
      }
    } catch (error) {
      console.error("Error unsubscribing from push notifications:", error)
      toast.error("Failed to disable push notifications")
    } finally {
      setIsLoading(false)
    }
  }, [isSupported])

  return {
    isSupported,
    isSubscribed,
    isLoading,
    permission,
    subscribe,
    unsubscribe,
    requestPermission,
  }
}
