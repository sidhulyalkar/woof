import * as SecureStore from 'expo-secure-store';
import apiClient, { ACCESS_TOKEN_KEY } from './client';

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
  await SecureStore.setItemAsync(ACCESS_TOKEN_KEY, response.access_token);
  return response;
}

function authHeader(token: string) {
  return { headers: { Authorization: `Bearer ${token}` } };
}

async function clearDeletedAccountCredentialBestEffort() {
  try {
    await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
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
    const response = await apiClient.post<AuthResponse>('/auth/register', data);
    return persist(response);
  },

  async login(data: LoginDto): Promise<AuthResponse> {
    const response = await apiClient.post<AuthResponse>('/auth/login', data);
    return persist(response);
  },

  async logout(): Promise<void> {
    const token = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
    await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
    if (!token) return;

    try {
      await apiClient.post('/auth/logout', {}, authHeader(token));
    } catch {
      // Local logout remains available when the server is unreachable or already
      // considers the captured session invalid.
    }
  },

  async logoutAll(): Promise<void> {
    const token = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
    await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
    if (!token) return;
    await apiClient.post('/auth/logout-all', {}, authHeader(token));
  },

  async deleteAccount(): Promise<void> {
    await apiClient.delete<void>('/users/me');
    await clearDeletedAccountCredentialBestEffort();
  },

  async getProfile() {
    return apiClient.get('/auth/me');
  },

  async isAuthenticated(): Promise<boolean> {
    return Boolean(await SecureStore.getItemAsync(ACCESS_TOKEN_KEY));
  },
};
