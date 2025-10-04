import { useSessionStore } from '@/store/session';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

interface ApiError {
  message: string;
  statusCode: number;
  error?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async refreshAccessToken(): Promise<string | null> {
    const { refreshToken, setTokens, logout } = useSessionStore.getState();

    if (!refreshToken) {
      return null;
    }

    try {
      const response = await fetch(`${this.baseUrl}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refreshToken }),
      });

      if (!response.ok) {
        logout();
        return null;
      }

      const data = await response.json();
      setTokens(data.accessToken, data.refreshToken);
      return data.accessToken;
    } catch (error) {
      logout();
      return null;
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const { accessToken } = useSessionStore.getState();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (accessToken) {
      headers['Authorization'] = `Bearer ${accessToken}`;
    }

    try {
      let response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers,
      });

      // If unauthorized, try to refresh the token
      if (response.status === 401 && accessToken) {
        const newAccessToken = await this.refreshAccessToken();

        if (newAccessToken) {
          headers['Authorization'] = `Bearer ${newAccessToken}`;
          response = await fetch(`${this.baseUrl}${endpoint}`, {
            ...options,
            headers,
          });
        }
      }

      if (!response.ok) {
        const error: ApiError = await response.json().catch(() => ({
          message: 'An error occurred',
          statusCode: response.status,
        }));

        throw new Error(error.message || `HTTP ${response.status}`);
      }

      // Handle empty responses
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }

      return {} as T;
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error');
    }
  }

  async get<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async put<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async patch<T>(endpoint: string, data?: any, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async delete<T>(endpoint: string, options?: RequestInit): Promise<T> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  async upload<T>(endpoint: string, formData: FormData, options?: RequestInit): Promise<T> {
    const { accessToken } = useSessionStore.getState();

    const headers: HeadersInit = {
      ...options?.headers,
    };

    if (accessToken) {
      headers['Authorization'] = `Bearer ${accessToken}`;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      method: 'POST',
      headers,
      body: formData,
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        message: 'Upload failed',
        statusCode: response.status,
      }));

      throw new Error(error.message || `HTTP ${response.status}`);
    }

    return await response.json();
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
