'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { trackScreenView, trackAppOpen, AnalyticsEvent, trackUserAction } from '@/lib/analytics';

export function useAnalytics() {
  const pathname = usePathname();

  useEffect(() => {
    trackAppOpen();
  }, []);

  useEffect(() => {
    if (pathname) {
      const screenName = pathname.split('/').filter(Boolean).join('_') || 'home';
      trackScreenView(screenName);
    }
  }, [pathname]);
}

export function useTrackAction() {
  return (event: AnalyticsEvent, metadata?: Record<string, unknown>) => {
    trackUserAction(event, metadata);
  };
}

export function useTrackComponent(componentName: string) {
  useEffect(() => {
    trackUserAction('SCREEN_VIEW', { component: componentName, action: 'mount' });

    return () => {
      trackUserAction('SCREEN_VIEW', { component: componentName, action: 'unmount' });
    };
  }, [componentName]);
}

export function useTrackFeature(featureName: string) {
  return (action: string, metadata?: Record<string, unknown>) => {
    trackUserAction('SCREEN_VIEW', {
      feature: featureName,
      action,
      ...metadata,
    });
  };
}
