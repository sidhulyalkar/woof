"use client"

import { useEffect } from "react"
import { usePathname } from "next/navigation"
import { trackScreenView, trackAppOpen, AnalyticsEvent, trackUserAction } from "@/lib/analytics"

/**
 * Hook to automatically track screen views on route changes
 */
export function useAnalytics() {
  const pathname = usePathname()

  useEffect(() => {
    // Track app open on first mount
    trackAppOpen()
  }, [])

  useEffect(() => {
    if (pathname) {
      const screenName = pathname.split("/").filter(Boolean).join("_") || "home"
      trackScreenView(screenName)
    }
  }, [pathname])
}

/**
 * Hook to track specific user actions
 */
export function useTrackAction() {
  return (event: AnalyticsEvent, metadata?: Record<string, any>) => {
    trackUserAction(event, metadata)
  }
}

/**
 * Hook to track component mount/unmount
 */
export function useTrackComponent(componentName: string) {
  useEffect(() => {
    trackUserAction("SCREEN_VIEW", { component: componentName, action: "mount" })

    return () => {
      trackUserAction("SCREEN_VIEW", { component: componentName, action: "unmount" })
    }
  }, [componentName])
}

/**
 * Hook to track feature usage
 */
export function useTrackFeature(featureName: string) {
  const trackFeatureUse = (action: string, metadata?: Record<string, any>) => {
    trackUserAction("SCREEN_VIEW", {
      feature: featureName,
      action,
      ...metadata,
    })
  }

  return trackFeatureUse
}
