import React, { ReactNode, createContext, useContext, useEffect, useState } from 'react';
import { AuthResponse, authApi } from '../api/auth';
import { User } from '../types';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, handle: string) => Promise<void>;
  logout: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const normalizeUser = (user: AuthResponse['user'] | any): User => ({
  ...user,
  displayName: user.displayName || user.handle,
});

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      if (await authApi.isAuthenticated()) {
        const profile = await authApi.getProfile();
        setUser(normalizeUser(profile));
      }
    } catch (error) {
      console.warn('Stored Woof session could not be restored', error);
      await authApi.logout();
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const response = await authApi.login({ email, password });
      setUser(normalizeUser(response.user));
    } finally {
      setLoading(false);
    }
  };

  const register = async (email: string, password: string, handle: string) => {
    setLoading(true);
    try {
      const response = await authApi.register({ email, password, handle });
      setUser(normalizeUser(response.user));
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      await authApi.logout();
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, isAuthenticated: Boolean(user) }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
