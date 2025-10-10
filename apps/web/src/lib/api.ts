import { apiClient } from './api/client';
import { useAuthStore } from './stores/auth-store';

// Authentication API calls
export const authApi = {
  /** Register a new user */
  register: async (data: { handle: string; email: string; password: string; bio?: string }) => {
    const response = await apiClient.post('/auth/register', data);
    // Auto-login after registration
    if (response.access_token && response.user) {
      useAuthStore.getState().setAuth(response.user, response.access_token);
    }
    return response;
  },

  /** Log in with email & password */
  login: async (data: { email: string; password: string }) => {
    const response = await apiClient.post('/auth/login', data);
    // Store auth data in store
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
  me: () => apiClient.get('/auth/me'),
};

// User-related API calls
export const userApi = {
  getUser: (userId: string) => apiClient.get(`/users/${userId}`),
};

// Pet-related API calls
export const petsApi = {
  getPets: () => apiClient.get('/pets'),
  createPet: (data: { name: string; species: string; [key: string]: any }) =>
    apiClient.post('/pets', data),
  getPet: (petId: string) => apiClient.get(`/pets/${petId}`),
};

// Activities (walk/run/play logs) API calls
export const activitiesApi = {
  getActivities: () => apiClient.get('/activities'),
  logActivity: (data: { petId: string; type: string; distance?: number; duration: number; calories?: number }) =>
    apiClient.post('/activities', data),
};

// Social (posts, feed, likes, comments) API calls
export const socialApi = {
  getFeed: () => apiClient.get('/social/posts'),
  createPost: (data: { text: string; mediaUrls?: string[]; petId?: string }) =>
    apiClient.post('/social/posts', data),
  likePost: (postId: string) => apiClient.post(`/social/posts/${postId}/likes`, {}),
  addComment: (postId: string, commentText: string) =>
    apiClient.post(`/social/posts/${postId}/comments`, { text: commentText }),
  getComments: (postId: string) => apiClient.get(`/social/posts/${postId}/comments`),
};

// Meetups (events) API calls
export const meetupsApi = {
  getMeetups: () => apiClient.get('/meetups'),
  createMeetup: (data: { title: string; datetime: string; location: string; description?: string }) =>
    apiClient.post('/meetups', data),
  rsvp: (meetupId: string, response: 'yes' | 'no' | 'maybe') =>
    apiClient.post(`/meetups/${meetupId}/rsvp`, { response }),
};

// Compatibility (pet matching) API calls
export const compatibilityApi = {
  getRecommendations: (petId: string) => apiClient.get(`/compatibility/recommendations/${petId}`),
  calculateCompatibility: (petId1: string, petId2: string) =>
    apiClient.post('/compatibility/calculate', { petId1, petId2 }),
};

// Events API calls
export const eventsApi = {
  getEvents: () => apiClient.get('/events'),
  getEvent: (eventId: string) => apiClient.get(`/events/${eventId}`),
  createEvent: (data: any) => apiClient.post('/events', data),
  updateEvent: (eventId: string, data: any) => apiClient.patch(`/events/${eventId}`, data),
  deleteEvent: (eventId: string) => apiClient.delete(`/events/${eventId}`),
  checkIn: (eventId: string) => apiClient.post(`/events/${eventId}/check-in`, {}),
};

// Gamification API calls
export const gamificationApi = {
  getProfile: (userId: string) => apiClient.get(`/gamification/profile/${userId}`),
  getLeaderboard: () => apiClient.get('/gamification/leaderboard'),
  awardPoints: (data: { userId: string; points: number; reason: string }) =>
    apiClient.post('/gamification/points', data),
};

// Services API calls
export const servicesApi = {
  getServices: (params?: any) => apiClient.get('/services', { params }),
  getService: (serviceId: string) => apiClient.get(`/services/${serviceId}`),
  trackIntent: (data: { serviceId: string; type: string }) =>
    apiClient.post('/services/intent', data),
};

// Verification API calls
export const verificationApi = {
  submitVerification: (data: any) => apiClient.post('/verification/submit', data),
  getStatus: () => apiClient.get('/verification/status'),
};

// Storage/Upload API calls
export const storageApi = {
  /** Upload a single file */
  uploadFile: async (file: File, folder?: string): Promise<{ key: string; url: string; bucket: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    if (folder) formData.append('folder', folder);

    const response = await apiClient.post('/storage/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  },

  /** Upload multiple files */
  uploadFiles: async (files: File[], folder?: string): Promise<Array<{ key: string; url: string; bucket: string }>> => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    if (folder) formData.append('folder', folder);

    const response = await apiClient.post('/storage/upload-multiple', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response;
  },

  /** Delete a file */
  deleteFile: (key: string) => apiClient.delete(`/storage/${key}`),
};

// Nudges API calls
export const nudgesApi = {
  /** Get active nudges for current user */
  getNudges: () => apiClient.get('/nudges'),

  /** Accept a nudge */
  acceptNudge: (nudgeId: string) => apiClient.patch(`/nudges/${nudgeId}/accept`, {}),

  /** Dismiss a nudge */
  dismissNudge: (nudgeId: string) => apiClient.patch(`/nudges/${nudgeId}/dismiss`, {}),

  /** Manually trigger chat activity check */
  checkChatActivity: (conversationId: string) =>
    apiClient.post(`/nudges/check/chat/${conversationId}`, {}),
};

// Push Notifications API calls
export const notificationsApi = {
  /** Subscribe to push notifications */
  subscribe: (subscription: any) => apiClient.post('/notifications/subscribe', { subscription }),

  /** Unsubscribe from push notifications */
  unsubscribe: () => apiClient.post('/notifications/unsubscribe', {}),

  /** Send a push notification (admin/testing) */
  sendPush: (data: { userId: string; title: string; body: string; url?: string; data?: any }) =>
    apiClient.post('/notifications/send', data),
};
