'use client';

import { Loader2 } from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import React, { useEffect, useRef, useState } from 'react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/stores/auth-store';

const PUBLIC_ROUTES = ['/login', '/onboarding', '/demo'];

function useCanonicalAuthHydration(isPublicRoute: boolean) {
  const [hasHydrated, setPersistenceHydrated] = useState(false);

  useEffect(() => {
    if (isPublicRoute) return;

    const stopHydrationListener = useAuthStore.persist.onHydrate(() => {
      setPersistenceHydrated(false);
    });
    const finishHydrationListener = useAuthStore.persist.onFinishHydration(() => {
      setPersistenceHydrated(true);
    });

    if (useAuthStore.persist.hasHydrated()) {
      setPersistenceHydrated(true);
    } else {
      // Persist middleware owns hydration truth. Explicitly request hydration as a
      // fallback so a protected production document cannot remain stranded if
      // automatic browser hydration has not completed before AuthGuard mounts.
      void useAuthStore.persist.rehydrate();
    }

    return () => {
      stopHydrationListener();
      finishHydrationListener();
    };
  }, [isPublicRoute]);

  return hasHydrated;
}

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isPublicRoute = PUBLIC_ROUTES.some((route) => pathname?.startsWith(route));
  const token = useAuthStore((state) => state.token);
  const hasHydrated = useCanonicalAuthHydration(isPublicRoute);
  const setAuth = useAuthStore((state) => state.setAuth);
  const logout = useAuthStore((state) => state.logout);
  const [isChecking, setIsChecking] = useState(true);
  const verifiedToken = useRef<string | null>(null);
  const verificationInFlight = useRef<string | null>(null);

  useEffect(() => {
    if (isPublicRoute) {
      verifiedToken.current = null;
      verificationInFlight.current = null;
      setIsChecking(false);
      return;
    }

    // Until Zustand's real persistence lifecycle completes, token === null means
    // "unknown", not "logged out". Keep protected content closed without
    // redirecting, then independently re-authorize any hydrated candidate token.
    if (!hasHydrated) {
      setIsChecking(true);
      return;
    }

    if (!token) {
      verifiedToken.current = null;
      verificationInFlight.current = null;
      setIsChecking(false);
      logout();
      router.replace('/login');
      return;
    }

    if (verifiedToken.current === token) {
      setIsChecking(false);
      return;
    }

    if (verificationInFlight.current === token) {
      return;
    }

    const candidateToken = token;
    let cancelled = false;
    verificationInFlight.current = candidateToken;
    setIsChecking(true);

    void authApi
      .me()
      .then((user) => {
        if (cancelled || useAuthStore.getState().token !== candidateToken) return;
        verifiedToken.current = candidateToken;
        setAuth(user, candidateToken);
      })
      .catch((error) => {
        if (cancelled || useAuthStore.getState().token !== candidateToken) return;
        console.error('Token verification failed:', error);
        verifiedToken.current = null;
        logout();
        router.replace('/login');
      })
      .finally(() => {
        if (cancelled) return;
        if (verificationInFlight.current === candidateToken) {
          verificationInFlight.current = null;
        }
        setIsChecking(false);
      });

    return () => {
      cancelled = true;
      if (verificationInFlight.current === candidateToken) {
        verificationInFlight.current = null;
      }
    };
  }, [hasHydrated, isPublicRoute, logout, router, setAuth, token]);

  // Public surfaces never need session verification. Rendering them synchronously
  // removes an unnecessary auth-spinner flash and keeps demos/login deterministic.
  if (isPublicRoute) {
    return <>{children}</>;
  }

  // A persisted bearer token is only a candidate credential after persistence has
  // hydrated and until /auth/me proves current server authority. Never expose
  // protected children during either window or while redirecting an invalid session.
  if (!hasHydrated || isChecking || !token || verifiedToken.current !== token) {
    if (hasHydrated && !token && !isChecking) return null;
    return (
      <div
        className="flex min-h-screen items-center justify-center"
        role="status"
        aria-live="polite"
      >
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
        <span className="sr-only">Checking your Woof session</span>
      </div>
    );
  }

  return <>{children}</>;
}
