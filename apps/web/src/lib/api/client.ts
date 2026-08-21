import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios';

type UnwrappedAxiosInstance = Omit<
  AxiosInstance,
  'request' | 'get' | 'delete' | 'head' | 'options' | 'post' | 'put' | 'patch'
> & {
  request<T = unknown, D = unknown>(config: AxiosRequestConfig<D>): Promise<T>;
  get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  head<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  options<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<T>;
  post<T = unknown, D = unknown>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<T>;
  put<T = unknown, D = unknown>(
    url: string,
    data?: D,
    config?: AxiosRequestConfig<D>
  ): Promise<T>;
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
    if (error.response?.status === 401) {
      console.warn('API responded with 401, logging out');
      if (typeof window !== 'undefined') {
        localStorage.removeItem('authToken');
        window.location.pathname = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export const apiClient = axiosClient as UnwrappedAxiosInstance;

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
