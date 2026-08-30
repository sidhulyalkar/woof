import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios';
import { useAuthStore } from '@/lib/stores/auth-store';

type UnwrappedAxiosInstance = Omit<
  AxiosInstance,
  'request' | 'get' | 'delete' | 'head' | 'options' | 'post' | 'put' | 'patch'
> & {
  request<T = unknown, D = unknown>(config: AxiosRequestConfig<D>): Promise<T>;
  get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  head<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  options<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  post<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<T>;
  put<T = unknown, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<T>;
  patch<T = unknown, D = unknown>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<T>;
  upload<T = unknown>(
    url: string,
    data: FormData,
    config?: AxiosRequestConfig<FormData>
  ): Promise<T>;
};

// Base API client for Woof API. The response interceptor below intentionally
// unwraps AxiosResponse.data, so the public method signatures return T rather
// than AxiosResponse<T>. Keeping the static contract aligned with runtime avoids
// transport details leaking throughout the product UI.
const axiosClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  withCredentials: false,
});

axiosClient.interceptors.request.use((config) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

axiosClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const auth = useAuthStore.getState();
    const requestHadAuthToken =
      typeof window !== 'undefined' &&
      Boolean(localStorage.getItem('authToken') || auth.token || auth.isAuthenticated);

    if (error.response?.status === 401 && requestHadAuthToken) {
      console.warn('Authenticated API request returned 401; clearing the stale session');
      auth.logout();
      if (window.location.pathname !== '/login') {
        window.location.pathname = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Axios' runtime interceptor intentionally changes the resolved value from
// AxiosResponse<T> to T. Cast through unknown so TypeScript treats this as an
// explicit transport-boundary adaptation rather than accidental structural overlap.
export const apiClient = axiosClient as unknown as UnwrappedAxiosInstance;

apiClient.upload = <T = unknown>(
  url: string,
  data: FormData,
  config?: AxiosRequestConfig<FormData>
) =>
  apiClient.post<T, FormData>(url, data, {
    ...config,
    headers: {
      'Content-Type': 'multipart/form-data',
      ...config?.headers,
    },
  });
