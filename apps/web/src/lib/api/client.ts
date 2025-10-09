import axios from 'axios';

// Base API client for Woof API
export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  withCredentials: false, // JWT will be sent in header, no cookies required
});

// Request interceptor to attach JWT token
apiClient.interceptors.request.use((config) => {
  const token = typeof window !== 'undefined' ? localStorage.getItem('authToken') : null;
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor to unwrap data and handle errors
apiClient.interceptors.response.use(
  response => {
    // Axios responses have the actual data in response.data
    // We can return response.data directly for simplicity
    return response.data;
  },
  error => {
    if (error.response) {
      // If we get a 401 Unauthorized, it means our token is invalid or expired
      if (error.response.status === 401) {
        console.warn('API responded with 401, logging out');
        localStorage.removeItem('authToken');
        // Optionally, redirect to login page
        if (typeof window !== 'undefined') {
          window.location.pathname = '/login';
        }
      }
    }
    return Promise.reject(error);
  }
);
