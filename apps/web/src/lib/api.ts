import { apiClient } from './api/client';
import { useAuthStore, type AuthPet, type AuthUser } from './stores/auth-store';
import type { Event, Match, ServiceProvider } from './types';

type AuthResponse = {
  access_token: string;
  user: AuthUser;
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

// Authentication API calls
export const authApi = {
  /** Register a new user */
  register: async (data: { handle: string; email: string; password: string; bio?: string }) => {
    const response = await apiClient.post<AuthResponse>('/auth/register', data);
    if (response.access_token && response.user) {
      useAuthStore.getState().setAuth(response.user, response.access_token);
    }
    return response;
  },

  /** Log in with email & password */
  login: async (data: { email: string; password: string }) => {
    const response = await apiClient.post<AuthResponse>('/auth/login', data);
    if (response.access_token && response.user) {
      useAuthStore.getState().setAuth(response.user, response.access_token);
    }
    return response;
  },

  /** Log out current user */
  logout: () => {
    useAuthStore.getState().logout();
  },

  /** Fetch current user profile (requires Authorization header) */
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

export const activitiesApi = {
  getActivities: () => apiClient.get<unknown[]>('/activities'),
  logActivity: (data: {
    petId: string;
    type: string;
    distance?: number;
    duration: number;
    calories?: number;
  }) => apiClient.post<unknown>('/activities', data),
};

export const socialApi = {
  getFeed: () => apiClient.get<unknown[]>('/social/posts'),
  createPost: (data: { text: string; mediaUrls?: string[]; petId?: string }) =>
    apiClient.post<unknown>('/social/posts', data),
  likePost: (postId: string) => apiClient.post<void>(`/social/posts/${postId}/likes`, {}),
  addComment: (postId: string, commentText: string) =>
    apiClient.post<unknown>(`/social/posts/${postId}/comments`, { text: commentText }),
  getComments: (postId: string) => apiClient.get<unknown[]>(`/social/posts/${postId}/comments`),
};

export const meetupsApi = {
  getMeetups: () => apiClient.get<unknown[]>('/meetups'),
  createMeetup: (data: {
    title: string;
    datetime: string;
    location: string;
    description?: string;
  }) => apiClient.post<unknown>('/meetups', data),
  rsvp: (meetupId: string, response: 'yes' | 'no' | 'maybe') =>
    apiClient.post<unknown>(`/meetups/${meetupId}/rsvp`, { response }),
};

type RawCompatibilityRecommendation = {
  id: string;
  pet: {
    id: string;
    ownerId: string;
    name: string;
    species: string;
    breed?: string | null;
    birthdate?: string | null;
    avatarUrl?: string | null;
    temperament?: string[];
    owner?: {
      id: string;
      handle: string;
      bio?: string | null;
      avatarUrl?: string | null;
      isVerified?: boolean;
    };
  };
  compatibilityScore: number;
  confidence?: number;
  source?: string;
  factors?: Record<string, number>;
  explanation?: string[];
  status?: 'PROPOSED' | 'CONFIRMED' | 'AVOID';
  lastInteractionAt?: string | null;
};

type RawRecommendationsResponse = {
  recommendations?: RawCompatibilityRecommendation[];
};

const asPercent = (value: number | undefined, fallback = 0) =>
  Math.round(Math.max(0, Math.min(1, value ?? fallback)) * 100);

const ageFromBirthdate = (birthdate?: string | null) => {
  if (!birthdate) return undefined;
  const born = new Date(birthdate);
  if (Number.isNaN(born.getTime())) return undefined;

  const now = new Date();
  let age = now.getFullYear() - born.getFullYear();
  const beforeBirthday =
    now.getMonth() < born.getMonth() ||
    (now.getMonth() === born.getMonth() && now.getDate() < born.getDate());
  if (beforeBirthday) age -= 1;
  return Math.max(0, age);
};

const normalizeRecommendation = (recommendation: RawCompatibilityRecommendation): Match => {
  const rawFactors = recommendation.factors ?? {};
  const owner = recommendation.pet.owner;

  return {
    id: recommendation.id,
    owner: {
      id: owner?.id ?? recommendation.pet.ownerId,
      name: owner?.handle ?? 'Woof member',
      bio: owner?.bio ?? undefined,
      avatarUrl: owner?.avatarUrl ?? undefined,
      isVerified: owner?.isVerified,
    },
    pet: {
      id: recommendation.pet.id,
      ownerId: recommendation.pet.ownerId,
      name: recommendation.pet.name,
      species: recommendation.pet.species,
      breed: recommendation.pet.breed ?? undefined,
      age: ageFromBirthdate(recommendation.pet.birthdate),
      temperament: recommendation.pet.temperament ?? [],
      photoUrl: recommendation.pet.avatarUrl ?? undefined,
    },
    compatibility: {
      overall: asPercent(recommendation.compatibilityScore),
      confidence: asPercent(recommendation.confidence, 0.5),
      source: recommendation.source ?? 'unknown',
      factors: {
        species: asPercent(rawFactors.species),
        ...(rawFactors.temperament !== undefined
          ? { temperament: asPercent(rawFactors.temperament) }
          : {}),
        ...(rawFactors.age !== undefined ? { age: asPercent(rawFactors.age) } : {}),
        ...(rawFactors.breed !== undefined ? { breed: asPercent(rawFactors.breed) } : {}),
      },
      explanation: recommendation.explanation ?? [],
    },
    status: recommendation.status,
    matchedAt: recommendation.lastInteractionAt ?? undefined,
  };
};

export const compatibilityApi = {
  /**
   * Normalize the API envelope into the single Match contract used by discovery.
   * This keeps presentation components independent from transport details.
   */
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

// Legacy read-only gamification surfaces remain for old profile screens. All
// reward writes are intentionally absent: Adventure rewards are server-owned
// CareEvents processed through RewardPolicy -> RewardLedger.
export const gamificationApi = {
  getProfile: (userId: string) => apiClient.get<unknown>(`/gamification/profile/${userId}`),
  getLeaderboard: () => apiClient.get<unknown[]>('/gamification/leaderboard'),
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
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  uploadFiles: async (files: File[], folder?: string): Promise<StorageObject[]> => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    if (folder) formData.append('folder', folder);

    return apiClient.post<StorageObject[]>('/storage/upload-multiple', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
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
