import { apiClient } from "./api-client"

export type AnalyticsEvent =
  | "APP_OPEN"
  | "APP_CLOSE"
  | "SCREEN_VIEW"
  | "MATCH_DISCOVERED"
  | "MEETUP_PROPOSED"
  | "MEETUP_ACCEPTED"
  | "MEETUP_DECLINED"
  | "EVENT_VIEWED"
  | "EVENT_RSVP"
  | "EVENT_CHECKIN"
  | "SERVICE_VIEWED"
  | "SERVICE_TAP_CALL"
  | "SERVICE_TAP_WEBSITE"
  | "SERVICE_TAP_BOOK"
  | "PROFILE_COMPLETED"
  | "PET_ADDED"
  | "QUIZ_STARTED"
  | "QUIZ_COMPLETED"
  | "CHAT_OPENED"
  | "MESSAGE_SENT"
  | "NUDGE_RECEIVED"
  | "NUDGE_ACCEPTED"
  | "NUDGE_DISMISSED"
  | "POINTS_EARNED"
  | "BADGE_EARNED"
  | "NOTIFICATION_ENABLED"
  | "NOTIFICATION_DISABLED"

interface TrackEventOptions {
  userId?: string
  metadata?: Record<string, any>
  source?: "WEB" | "IOS" | "ANDROID"
}

/**
 * Track an analytics event
 */
export async function trackEvent(
  event: AnalyticsEvent,
  options?: TrackEventOptions,
): Promise<void> {
  try {
    await apiClient.post("/analytics/telemetry", {
      userId: options?.userId,
      source: options?.source || "WEB",
      event,
      metadata: options?.metadata || {},
    })
  } catch (error) {
    // Silently fail - don't disrupt user experience
    console.error("[Analytics] Failed to track event:", event, error)
  }
}

/**
 * Track screen view
 */
export function trackScreenView(screenName: string, metadata?: Record<string, any>): void {
  trackEvent("SCREEN_VIEW", {
    metadata: {
      screen: screenName,
      ...metadata,
    },
  })
}

/**
 * Track app open (call on mount)
 */
export function trackAppOpen(): void {
  trackEvent("APP_OPEN", {
    metadata: {
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
      },
    },
  })
}

/**
 * Track user action with automatic userId from session
 */
export function trackUserAction(
  event: AnalyticsEvent,
  metadata?: Record<string, any>,
): void {
  // Get userId from session storage or auth store
  const userId = localStorage.getItem("userId") || undefined

  trackEvent(event, {
    userId,
    metadata,
  })
}

/**
 * Batch tracking for multiple events
 */
export async function trackBatch(
  events: Array<{ event: AnalyticsEvent; metadata?: Record<string, any> }>,
): Promise<void> {
  try {
    await Promise.all(events.map((e) => trackEvent(e.event, { metadata: e.metadata })))
  } catch (error) {
    console.error("[Analytics] Failed to track batch events:", error)
  }
}

/**
 * Hook for tracking page views in Next.js
 */
export function usePageTracking() {
  if (typeof window === "undefined") return

  const trackPageView = () => {
    const path = window.location.pathname
    const screenName = path.split("/").filter(Boolean).join("_") || "home"
    trackScreenView(screenName)
  }

  // Track initial page view
  trackPageView()

  // Track route changes (for client-side navigation)
  const originalPushState = history.pushState
  const originalReplaceState = history.replaceState

  history.pushState = function (...args) {
    originalPushState.apply(this, args)
    trackPageView()
  }

  history.replaceState = function (...args) {
    originalReplaceState.apply(this, args)
    trackPageView()
  }

  window.addEventListener("popstate", trackPageView)

  return () => {
    history.pushState = originalPushState
    history.replaceState = originalReplaceState
    window.removeEventListener("popstate", trackPageView)
  }
}

/**
 * Performance tracking
 */
export function trackPerformance(metricName: string, value: number): void {
  trackEvent("SCREEN_VIEW", {
    metadata: {
      metric: metricName,
      value,
      timestamp: Date.now(),
    },
  })
}

/**
 * Error tracking
 */
export function trackError(error: Error, context?: Record<string, any>): void {
  trackEvent("APP_CLOSE", {
    metadata: {
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name,
      },
      context,
    },
  })
}
