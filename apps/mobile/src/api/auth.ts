import apiClient from './client';
import {
  clearAccessToken,
  clearAccessTokenIfCurrent,
  getAccessToken,
  storeAccessToken,
  takeAccessToken,
} from './session';
import { clearRegistrationRecovery, getOrCreateRegistrationRecovery } from '../onboarding/recovery';

export interface RegisterDto {
  email: string;
  password: string;
  handle: string;
  bio?: string;
}

export interface LoginDto {
  email: string;
  password: string;
}

export interface AuthUser {
  id: string;
  email: string;
  handle: string;
  bio?: string | null;
  avatarUrl?: string | null;
}

export interface AuthResponse {
  access_token: string;
  user: AuthUser;
}

async function persist(response: AuthResponse) {
  await storeAccessToken(response.access_token);
  return response;
}

function authHeader(token: string) {
  return { headers: { Authorization: `Bearer ${token}` } };
}

async function clearDeletedAccountCredentialBestEffort() {
  try {
    await clearAccessToken();
  } catch (error) {
    // Server deletion is already authoritative and removes the canonical session.
    // Do not turn a successful account deletion into a false client failure if
    // the device credential store cannot be cleaned immediately. Any retained
    // token is rejected by server-side session authority and cleared on a later
    // 401/session restoration attempt.
    console.warn('Woof account was deleted, but local credential cleanup must retry', error);
  }
}

export const authApi = {
  async register(data: RegisterDto): Promise<AuthResponse> {
    const recovery = await getOrCreateRegistrationRecovery(data.email, data.handle);
    const response = await apiClient.post<AuthResponse>('/auth/register', {
      ...data,
      email: data.email.trim().toLowerCase(),
      handle: data.handle.trim().toLowerCase(),
      registrationKey: recovery.registrationKey,
    });

    // Keep the replay key until both server registration and local credential
    // persistence succeed. If either response edge is lost, the next exact
    // retry can converge on the same canonical account and request a fresh
    // server-owned session.
    const persisted = await persist(response);
    await clearRegistrationRecovery();
    return persisted;
  },

  async login(data: LoginDto): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>('/auth/login', data);
    return persist(response);
  },

  async logout(): Promise<void> {
    const token = await takeAccessToken();
    if (!token) return;

    try {
      await apiClient.post('/auth/logout', {}, authHeader(token));
    } catch {
      // Local logout remains available when the server is unreachable or already
      // considers the captured session invalid.
    }
  },

  async logoutAll(): Promise<void> {
    const token = await getAccessToken();
    if (!token) return;

    // "All devices" is a server-owned claim. Keep this device credential until
    // the server confirms every active session for the user has been revoked so
    // a transient outage remains retryable instead of silently becoming a local-only logout.
    await apiClient.post('/auth/logout-all', {}, authHeader(token));
    const cleanup = await clearAccessTokenIfCurrent(token);
    if (cleanup.matched && !cleanup.cleared) {
      console.warn('All Woof sessions were revoked, but local credential cleanup must retry');
    }
  },

  async deleteAccount(): Promise<void> {
    await apiClient.delete<void>('/users/me');
    await clearDeletedAccountCredentialBestEffort();
  },

  async getProfile() {
    return apiClient.get('/auth/me');
  },

  async isAuthenticated(): Promise<boolean> {
    return Boolean(await getAccessToken());
  },
};
