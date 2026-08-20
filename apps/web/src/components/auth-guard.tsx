'use client';

import React, { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';
import { authApi } from '@/lib/api';
import { useAuthStore } from '@/lib/stores/auth-store';

const PUBLIC_ROUTES = ['/login', '/onboarding', '/demo'];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, token, setAuth, setLoading, logout } = useAuthStore();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    async function checkAuth() {
      const isPublicRoute = PUBLIC_ROUTES.some((route) => pathname?.startsWith(route));

      if (isPublicRoute) {
        setIsChecking(false);
        return;
      }

      const storedToken = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;

      if (!isAuthenticated && !storedToken) {
        setIsChecking(false);
        router.replace('/login');
        return;
      }

      if (storedToken && !isAuthenticated) {
        try {
          setLoading(true);
          const user = await authApi.me();
          setAuth(user, storedToken);
        } catch (error) {
          console.error('Token verification failed:', error);
          logout();
          router.replace('/login');
        } finally {
          setLoading(false);
        }
      }

      setIsChecking(false);
    }

    void checkAuth();
  }, [isAuthenticated, token, pathname, router, setAuth, setLoading, logout]);

  if (isChecking) {
    return (
      <div className="flex min-h-screen items-center justify-center" role="status" aria-live="polite">
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
        <span className="sr-only">Checking your Woof session</span>
      </div>
    );
  }

  return <>{children}</>;
}
