import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:4000/api/v1';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear token and redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/auth/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authApi = {
  register: async (data: { email: string; password: string; handle: string; bio?: string }) => {
    const response = await api.post('/auth/register', data);
    if (response.data.access_token) {
      localStorage.setItem('auth_token', response.data.access_token);
    }
    return response.data;
  },

  login: async (data: { email: string; password: string }) => {
    const response = await api.post('/auth/login', data);
    if (response.data.access_token) {
      localStorage.setItem('auth_token', response.data.access_token);
    }
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('auth_token');
  },

  getProfile: async () => {
    const response = await api.get('/auth/profile');
    return response.data;
  },
};

// Users API
export const usersApi = {
  getUser: async (id: string) => {
    const response = await api.get(`/users/${id}`);
    return response.data;
  },

  updateUser: async (id: string, data: Partial<{ bio: string; avatarUrl: string }>) => {
    const response = await api.patch(`/users/${id}`, data);
    return response.data;
  },

  getUserPets: async (id: string) => {
    const response = await api.get(`/users/${id}/pets`);
    return response.data;
  },
};

// Pets API
export const petsApi = {
  getAllPets: async () => {
    const response = await api.get('/pets');
    return response.data;
  },

  createPet: async (data: {
    name: string;
    species: string;
    breed?: string;
    age?: number;
    bio?: string;
    avatarUrl?: string;
  }) => {
    const response = await api.post('/pets', data);
    return response.data;
  },

  getPet: async (id: string) => {
    const response = await api.get(`/pets/${id}`);
    return response.data;
  },

  updatePet: async (id: string, data: Partial<{
    name: string;
    breed: string;
    age: number;
    bio: string;
    avatarUrl: string;
  }>) => {
    const response = await api.patch(`/pets/${id}`, data);
    return response.data;
  },

  deletePet: async (id: string) => {
    const response = await api.delete(`/pets/${id}`);
    return response.data;
  },
};

// Activities API
export const activitiesApi = {
  getAllActivities: async () => {
    const response = await api.get('/activities');
    return response.data;
  },

  createActivity: async (data: {
    petId: string;
    type: string;
    duration?: number;
    distance?: number;
    caloriesBurned?: number;
    notes?: string;
  }) => {
    const response = await api.post('/activities', data);
    return response.data;
  },

  getPetActivities: async (petId: string) => {
    const response = await api.get(`/activities/pet/${petId}`);
    return response.data;
  },
};

// Social API
export const socialApi = {
  getFeed: async () => {
    const response = await api.get('/social/feed');
    return response.data;
  },

  createPost: async (data: {
    petId: string;
    content: string;
    imageUrl?: string;
    location?: string;
  }) => {
    const response = await api.post('/social/posts', data);
    return response.data;
  },

  likePost: async (postId: string) => {
    const response = await api.post(`/social/posts/${postId}/like`);
    return response.data;
  },

  unlikePost: async (postId: string) => {
    const response = await api.delete(`/social/posts/${postId}/like`);
    return response.data;
  },

  commentOnPost: async (postId: string, content: string) => {
    const response = await api.post(`/social/posts/${postId}/comments`, { content });
    return response.data;
  },

  getLeaderboard: async () => {
    const response = await api.get('/social/leaderboard');
    return response.data;
  },
};

// Meetups API
export const meetupsApi = {
  getAllMeetups: async () => {
    const response = await api.get('/meetups');
    return response.data;
  },

  createMeetup: async (data: {
    title: string;
    description?: string;
    location: string;
    scheduledFor: string;
    maxParticipants?: number;
  }) => {
    const response = await api.post('/meetups', data);
    return response.data;
  },

  joinMeetup: async (meetupId: string) => {
    const response = await api.post(`/meetups/${meetupId}/join`);
    return response.data;
  },

  leaveMeetup: async (meetupId: string) => {
    const response = await api.delete(`/meetups/${meetupId}/join`);
    return response.data;
  },
};

// Compatibility API
export const compatibilityApi = {
  getCompatiblePets: async (petId: string) => {
    const response = await api.get(`/compatibility/${petId}/compatible-pets`);
    return response.data;
  },

  findPlaymates: async (petId: string) => {
    const response = await api.get(`/compatibility/${petId}/find-playmates`);
    return response.data;
  },
};
