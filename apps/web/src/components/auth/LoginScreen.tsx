'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { apiClient } from '@/lib/api/client';
import { useSessionStore } from '@/store/session';
import { useUIStore } from '@/store/ui';
import { Loader2 } from 'lucide-react';

export function LoginScreen() {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [handle, setHandle] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login } = useSessionStore();
  const { showToast } = useUIStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      if (isLogin) {
        // Login
        const response = await apiClient.post<any>('/auth/login', {
          email,
          password,
        });

        login(response.user, response.access_token, response.access_token);
        showToast({ message: 'Logged in successfully!', type: 'success' });
      } else {
        // Register
        const response = await apiClient.post<any>('/auth/register', {
          email,
          password,
          handle,
        });

        login(response.user, response.access_token, response.access_token);
        showToast({ message: 'Account created successfully!', type: 'success' });
      }
    } catch (error: any) {
      showToast({
        message: error.message || 'Authentication failed',
        type: 'error'
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background">
      <div className="w-full max-w-md glass-card rounded-3xl p-8 space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent mb-2">
            Woof
          </h1>
          <p className="text-muted-foreground">
            {isLogin ? 'Welcome back!' : 'Join the pack!'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <div>
              <label className="text-sm font-medium mb-2 block">Username</label>
              <Input
                type="text"
                value={handle}
                onChange={(e) => setHandle(e.target.value)}
                placeholder="buddy_owner"
                required
                className="bg-background/50"
              />
            </div>
          )}

          <div>
            <label className="text-sm font-medium mb-2 block">Email</label>
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              className="bg-background/50"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Password</label>
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              className="bg-background/50"
            />
          </div>

          <Button
            type="submit"
            disabled={isLoading}
            className="w-full h-12 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 font-semibold"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                {isLogin ? 'Logging in...' : 'Creating account...'}
              </>
            ) : (
              isLogin ? 'Log In' : 'Sign Up'
            )}
          </Button>
        </form>

        <div className="text-center">
          <button
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Log in'}
          </button>
        </div>

        {/* Quick test account hint */}
        <div className="text-center pt-4 border-t border-border/20">
          <p className="text-xs text-muted-foreground">
            Test account: test@woof.com / password123
          </p>
        </div>
      </div>
    </div>
  );
}
