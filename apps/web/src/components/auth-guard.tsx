"use client";

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuthStore } from '@/lib/stores/auth-store';
import { authApi } from '@/lib/api';
import { Loader2 } from 'lucide-react';

const PUBLIC_ROUTES = ['/login', '/onboarding'];

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { isAuthenticated, token, setAuth, setLoading, logout } = useAuthStore();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    async function checkAuth() {
      // Check if current route is public
      const isPublicRoute = PUBLIC_ROUTES.some(route => pathname?.startsWith(route));

      if (isPublicRoute) {
        setIsChecking(false);
        return; // Allow access to public routes
      }

      // Check for token
      const storedToken = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;

      if (!isAuthenticated && !storedToken) {
        // No token found, redirect to login
        setIsChecking(false);
        router.replace('/login');
        return;
      }

      // If we have a token but not authenticated in store, verify it
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

    checkAuth();
  }, [isAuthenticated, token, pathname, router, setAuth, setLoading, logout]);

  // Show loading spinner while checking auth
  if (isChecking) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return <>{children}</>;
}
