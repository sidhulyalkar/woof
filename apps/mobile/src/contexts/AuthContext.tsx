import React, {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import { AuthResponse, authApi } from '../api/auth';
import { getAccessToken, subscribeToSessionInvalidation } from '../api/session';
import { User } from '../types';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  sessionVerificationUnavailable: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, handle: string) => Promise<void>;
  logout: () => Promise<void>;
  logoutAll: () => Promise<void>;
  retrySessionVerification: () => Promise<void>;
  deleteAccount: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const normalizeUser = (user: AuthResponse['user'] | any): User => ({
  ...user,
  displayName: user.displayName || user.handle,
});

function isUnauthorizedResponse(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const response = (error as { response?: { status?: unknown } }).response;
  return response?.status === 401;
}

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionVerificationUnavailable, setSessionVerificationUnavailable] = useState(false);
  const activeTokenRef = useRef<string | null>(null);

  const restoreSession = useCallback(async () => {
    setLoading(true);
    try {
      const storedToken = await getAccessToken();
      activeTokenRef.current = storedToken;

      if (!storedToken) {
        setUser(null);
        setSessionVerificationUnavailable(false);
        return;
      }

      try {
        const profile = await authApi.getProfile();
        if (activeTokenRef.current !== storedToken) return;
        setUser(normalizeUser(profile));
        setSessionVerificationUnavailable(false);
      } catch (error) {
        if (activeTokenRef.current !== storedToken) return;

        if (isUnauthorizedResponse(error)) {
          activeTokenRef.current = null;
          setUser(null);
          setSessionVerificationUnavailable(false);
          return;
        }

        // A timeout/offline/5xx response does not prove the persisted server
        // session is invalid. Preserve the credential and close authenticated
        // product surfaces until authority can be checked again.
        console.warn('Stored Woof session could not be verified yet', error);
        setUser(null);
        setSessionVerificationUnavailable(true);
      }
    } catch (error) {
      // Secure credential access itself is also an authority failure. Do not
      // guess that the user is signed in or erase a credential we could not read.
      console.warn('Woof could not read the stored session credential', error);
      activeTokenRef.current = null;
      setUser(null);
      setSessionVerificationUnavailable(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const unsubscribe = subscribeToSessionInvalidation((event) => {
      // The API layer publishes the exact rejected bearer token. This in-memory
      // identity check prevents a delayed 401 from an older request from
      // evicting a newer session that has already replaced it.
      if (activeTokenRef.current !== event.rejectedToken) return;
      activeTokenRef.current = null;
      setUser(null);
      setSessionVerificationUnavailable(false);
      setLoading(false);
    });

    void restoreSession();
    return unsubscribe;
  }, [restoreSession]);

  const login = async (email: string, password: string) => {
    setLoading(true);
    setSessionVerificationUnavailable(false);
    try {
      const response = await authApi.login({ email, password });
      activeTokenRef.current = response.access_token;
      setUser(normalizeUser(response.user));
    } finally {
      setLoading(false);
    }
  };

  const register = async (email: string, password: string, handle: string) => {
    setLoading(true);
    setSessionVerificationUnavailable(false);
    try {
      const response = await authApi.register({ email, password, handle });
      activeTokenRef.current = response.access_token;
      setUser(normalizeUser(response.user));
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      await authApi.logout();
      activeTokenRef.current = null;
      setUser(null);
      setSessionVerificationUnavailable(false);
    } finally {
      setLoading(false);
    }
  };

  const logoutAll = async () => {
    setLoading(true);
    try {
      await authApi.logoutAll();
      activeTokenRef.current = null;
      setUser(null);
      setSessionVerificationUnavailable(false);
    } finally {
      setLoading(false);
    }
  };

  const deleteAccount = async () => {
    setLoading(true);
    try {
      await authApi.deleteAccount();
      activeTokenRef.current = null;
      setUser(null);
      setSessionVerificationUnavailable(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        sessionVerificationUnavailable,
        login,
        register,
        logout,
        logoutAll,
        retrySessionVerification: restoreSession,
        deleteAccount,
        isAuthenticated: Boolean(user),
      }}
    >
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
