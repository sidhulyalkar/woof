'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/stores/auth-store';

const PUBLIC_ROUTES = ['/login', '/onboarding', '/demo'];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const setAuth = useAuthStore((state) => state.setAuth);
  const logout = useAuthStore((state) => state.logout);
  const [isChecking, setIsChecking] = useState(true);
  const hydrationInFlight = useRef(false);

  useEffect(() => {
    const isPublicRoute = PUBLIC_ROUTES.some((route) => pathname?.startsWith(route));

    if (isPublicRoute) {
      hydrationInFlight.current = false;
      setIsChecking(false);
      return;
    }

    if (isAuthenticated) {
      hydrationInFlight.current = false;
      setIsChecking(false);
      return;
    }

    const storedToken =
      typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;

    if (!storedToken) {
      hydrationInFlight.current = false;
      setIsChecking(false);
      router.replace('/login');
      return;
    }

    if (hydrationInFlight.current) {
      return;
    }

    hydrationInFlight.current = true;
    setIsChecking(true);

    void authApi
      .me()
      .then((user) => {
        setAuth(user, storedToken);
      })
      .catch((error) => {
        console.error('Token verification failed:', error);
        logout();
        router.replace('/login');
      })
      .finally(() => {
        hydrationInFlight.current = false;
        setIsChecking(false);
      });
  }, [isAuthenticated, pathname, router, setAuth, logout]);

  if (isChecking) {
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
