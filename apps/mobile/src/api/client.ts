import { create, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios';
import Constants from 'expo-constants';
import * as SecureStore from 'expo-secure-store';

const DEVELOPMENT_API_URL = 'http://localhost:4000/api/v1';
const NON_REMOTE_HOSTS = new Set(['localhost', '127.0.0.1', '0.0.0.0', '::1']);
const configuredBuildProfile = Constants.expoConfig?.extra?.buildProfile;
const BUILD_PROFILE =
  typeof configuredBuildProfile === 'string' && configuredBuildProfile.trim()
    ? configuredBuildProfile.trim()
    : 'development';
const configuredApiUrl =
  process.env.EXPO_PUBLIC_API_URL || Constants.expoConfig?.extra?.apiUrl || DEVELOPMENT_API_URL;

function validateApiUrl(value: string): string {
  const normalized = value.replace(/\/$/, '');
  if (BUILD_PROFILE === 'development') return normalized;

  let parsed: URL;
  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error(`Woof ${BUILD_PROFILE} API URL is not a valid absolute URL`);
  }

  if (parsed.protocol !== 'https:' || NON_REMOTE_HOSTS.has(parsed.hostname.toLowerCase())) {
    throw new Error(`Woof ${BUILD_PROFILE} API URL must be a remote HTTPS endpoint`);
  }

  return normalized;
}

const API_URL = validateApiUrl(configuredApiUrl);
const ACCESS_TOKEN_KEY = 'woofAccessToken';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = create({
      baseURL: API_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.client.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        const token = await SecureStore.getItemAsync(ACCESS_TOKEN_KEY);
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // The canonical NestJS API currently issues one expiring access token and
    // does not expose a refresh-token endpoint. Do not invent a client-side
    // protocol that the server cannot honor. Clear stale credentials on 401 so
    // navigation/session code can return the user to authentication cleanly.
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config = {}) {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: unknown, config = {}) {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: unknown, config = {}) {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: unknown, config = {}) {
    const response = await this.client.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config = {}) {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

export { ACCESS_TOKEN_KEY, API_URL, BUILD_PROFILE };
const apiClient = new ApiClient();
export default apiClient;
