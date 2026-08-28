import { apiClient } from './api/client';
import { disconnectSocket } from './socket';
import { useAuthStore, type AuthPet, type AuthUser } from './stores/auth-store';
import type { Event, Match, ServiceProvider } from './types';

type AuthResponse = {
  access_token: string;
  user: AuthUser;
};

type RegistrationRequest = {
  handle: string;
  email: string;
  password: string;
  bio?: string;
  registrationKey?: string;
};

type StorageObject = {
  key: string;
  url: string;
  bucket: string;
};

export type Nudge = {
  id: string;
  type: 'meetup' | 'service' | 'event' | 'achievement';
  payload: {
    targetUserId?: string;
    reason: 'proximity' | 'chat_activity' | 'mutual_availability' | 'goal_achievement';
    message?: string;
    location?: { lat: number; lng: number };
    metadata?: Record<string, unknown>;
  };
  createdAt: string;
  dismissed: boolean;
};

function clearLocalAuth() {
  disconnectSocket();
  useAuthStore.getState().logout();
}

function authHeader(token: string) {
  return { headers: { Authorization: `Bearer ${token}` } };
}

export const authApi = {
  register: async (data: RegistrationRequest) => {
    const response = await apiClient.post<AuthResponse>('/auth/register', data);
    if (response.access_token && response.user) {
      useAuthStore.getState().setAuth(response.user, response.access_token);
    }
    return response;
  },

  login: async (data: { email: string; password: string }) => {
    const response = await apiClient.post<AuthResponse>('/auth/login', data);
    if (response.access_token && response.user) {
      useAuthStore.getState().setAuth(response.user, response.access_token);
    }
    return response;
  },

  logout: async () => {
    const token = useAuthStore.getState().token;
    clearLocalAuth();
    if (!token) return;

    try {
      await apiClient.post('/auth/logout', {}, authHeader(token));
    } catch {
      // Local logout is authoritative for this device when the server is unreachable
      // or already considers the captured session invalid.
    }
  },

  logoutAll: async () => {
    const token = useAuthStore.getState().token;
    clearLocalAuth();
    if (!token) return;
    await apiClient.post('/auth/logout-all', {}, authHeader(token));
  },

  me: () => apiClient.get<AuthUser>('/auth/me'),
};

export const userApi = {
  getUser: (userId: string) => apiClient.get<AuthUser>(`/users/${userId}`),
};

export const petsApi = {
  getPets: () => apiClient.get<AuthPet[]>('/pets'),
  createPet: (data: { name: string; species: string; [key: string]: unknown }) =>
    apiClient.post<AuthPet>('/pets', data),
  getPet: (petId: string) => apiClient.get<AuthPet>(`/pets/${petId}`),
};

export const postsApi = {
  getFeed: () => apiClient.get<unknown[]>('/social/feed'),
  createPost: (data: { content?: string; [key: string]: unknown }) =>
    apiClient.post<unknown>('/social/posts', data),
  likePost: (postId: string) => apiClient.post<void>(`/social/posts/${postId}/like`, {}),
  unlikePost: (postId: string) => apiClient.delete<void>(`/social/posts/${postId}/like`),
};

export const compatibilityApi = {
  getRecommendations: async (petId: string): Promise<Match[]> => {
    const response = await apiClient.get<
      RawRecommendationsResponse | RawCompatibilityRecommendation[]
    >(`/compatibility/recommendations/${petId}`);

    const recommendations = Array.isArray(response) ? response : (response.recommendations ?? []);
    return recommendations.map(normalizeRecommendation);
  },

  calculateCompatibility: (petAId: string, petBId: string) =>
    apiClient.post<unknown>('/compatibility/calculate', { petAId, petBId }),
};

export const eventsApi = {
  getEvents: () => apiClient.get<Event[]>('/events'),
  getEvent: (eventId: string) => apiClient.get<Event>(`/events/${eventId}`),
  createEvent: (data: Partial<Event>) => apiClient.post<Event>('/events', data),
  updateEvent: (eventId: string, data: Partial<Event>) =>
    apiClient.patch<Event>(`/events/${eventId}`, data),
  deleteEvent: (eventId: string) => apiClient.delete<void>(`/events/${eventId}`),
  checkIn: (eventId: string) => apiClient.post<unknown>(`/events/${eventId}/check-in`, {}),
};

export const gamificationApi = {
  getProfile: (userId: string) => apiClient.get<unknown>(`/gamification/profile/${userId}`),
  getLeaderboard: () => apiClient.get<unknown[]>('/gamification/leaderboard'),
  awardPoints: (data: { userId: string; points: number; reason: string }) =>
    apiClient.post<unknown>('/gamification/points', data),
};

export const servicesApi = {
  getServices: (params?: Record<string, unknown>) =>
    apiClient.get<ServiceProvider[]>('/services', { params }),
  getService: (serviceId: string) => apiClient.get<ServiceProvider>(`/services/${serviceId}`),
  trackIntent: (data: { serviceId: string; type: string }) =>
    apiClient.post<void>('/services/intent', data),
};

export const verificationApi = {
  submitVerification: (data: Record<string, unknown>) =>
    apiClient.post<unknown>('/verification/submit', data),
  getStatus: () => apiClient.get<unknown>('/verification/status'),
};

export const storageApi = {
  uploadFile: async (file: File, folder?: string): Promise<StorageObject> => {
    const formData = new FormData();
    formData.append('file', file);
    if (folder) formData.append('folder', folder);
    return apiClient.post<StorageObject>('/storage/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  uploadFiles: async (files: File[], folder?: string): Promise<StorageObject[]> => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    if (folder) formData.append('folder', folder);
    return apiClient.post<StorageObject[]>('/storage/upload-multiple', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  deleteFile: (key: string) => apiClient.delete<void>(`/storage/${key}`),
};

export const nudgesApi = {
  getNudges: () => apiClient.get<Nudge[]>('/nudges'),
  acceptNudge: (nudgeId: string) => apiClient.patch<void>(`/nudges/${nudgeId}/accept`, {}),
  dismissNudge: (nudgeId: string) => apiClient.patch<void>(`/nudges/${nudgeId}/dismiss`, {}),
  checkChatActivity: (conversationId: string) =>
    apiClient.post<unknown>(`/nudges/check/chat/${conversationId}`, {}),
};

export const notificationsApi = {
  subscribe: (subscription: PushSubscriptionJSON) =>
    apiClient.post<void>('/notifications/subscribe', { subscription }),
  unsubscribe: () => apiClient.post<void>('/notifications/unsubscribe', {}),
  sendPush: (data: {
    userId: string;
    title: string;
    body: string;
    url?: string;
    data?: Record<string, unknown>;
  }) => apiClient.post<void>('/notifications/send', data),
};

export const analyticsApi = {
  trackEvent: (data: {
    userId?: string;
    source: string;
    event: string;
    metadata?: Record<string, unknown>;
  }) => apiClient.post<void>('/analytics/telemetry', data),
  getNorthStar: (timeframe?: '7d' | '30d' | '90d') =>
    apiClient.get<unknown>('/analytics/north-star', { params: { timeframe } }),
  getDetails: (timeframe?: '7d' | '30d' | '90d') =>
    apiClient.get<unknown>('/analytics/details', { params: { timeframe } }),
  getEventCounts: (timeframe?: '7d' | '30d' | '90d') =>
    apiClient.get<unknown>('/analytics/events', { params: { timeframe } }),
  getActiveUsers: (timeframe?: '7d' | '30d' | '90d') =>
    apiClient.get<unknown>('/analytics/users/active', { params: { timeframe } }),
  getScreenViews: (timeframe?: '7d' | '30d' | '90d') =>
    apiClient.get<unknown>('/analytics/screens', { params: { timeframe } }),
  getUserActivity: (userId: string, limit?: number) =>
    apiClient.get<unknown>(`/analytics/users/${userId}/activity`, { params: { limit } }),
};
