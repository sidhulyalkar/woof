// Shared types for the mobile app

export interface User {
  id: string;
  email: string;
  handle: string;
  displayName: string;
  bio?: string;
  location?: string;
  avatarUrl?: string;
  createdAt: string;
  points?: number;
  level?: number;
}

export interface Pet {
  id: string;
  name: string;
  species: 'dog' | 'cat' | 'other';
  breed?: string;
  age?: number;
  weight?: number;
  gender?: 'male' | 'female' | 'unknown';
  bio?: string;
  avatarUrl?: string;
  energyLevel?: number;
  playStyle?: string[];
  temperament?: string[];
  ownerId: string;
  owner?: User;
  createdAt: string;
}

export interface Activity {
  id: string;
  type: 'walk' | 'run' | 'play' | 'training' | 'other';
  title: string;
  description?: string;
  duration: number; // minutes
  distance?: number; // meters
  calories?: number;
  startTime: string;
  endTime?: string;
  location?: {
    latitude: number;
    longitude: number;
    address?: string;
  };
  petId: string;
  pet?: Pet;
  userId: string;
  user?: User;
  photos?: string[];
  coActivityId?: string;
  createdAt: string;
}

export interface Post {
  id: string;
  content: string;
  mediaUrls?: string[];
  userId: string;
  user?: User;
  petId?: string;
  pet?: Pet;
  activityId?: string;
  activity?: Activity;
  likesCount: number;
  commentsCount: number;
  isLiked?: boolean;
  createdAt: string;
}

export interface Comment {
  id: string;
  content: string;
  postId: string;
  userId: string;
  user?: User;
  createdAt: string;
}

export interface CommunityEvent {
  id: string;
  title: string;
  description: string;
  startTime: string;
  endTime: string;
  location: {
    latitude: number;
    longitude: number;
    address: string;
  };
  capacity?: number;
  hostUserId: string;
  host?: User;
  rsvpsCount: number;
  hasRsvped?: boolean;
  attendeesCount?: number;
  status: 'upcoming' | 'ongoing' | 'completed' | 'cancelled';
  imageUrl?: string;
  createdAt: string;
}

export interface EventRSVP {
  id: string;
  eventId: string;
  userId: string;
  status: 'going' | 'maybe' | 'not_going';
  petIds?: string[];
  checkedInAt?: string;
  createdAt: string;
}

export interface ChatConversation {
  id: string;
  participants: User[];
  lastMessage?: ChatMessage;
  unreadCount: number;
  updatedAt: string;
}

export interface ChatMessage {
  id: string;
  conversationId: string;
  senderId: string;
  sender?: User;
  content: string;
  type: 'text' | 'image' | 'location';
  mediaUrl?: string;
  createdAt: string;
  readBy: string[];
}

export interface Nudge {
  id: string;
  userId: string;
  type: 'walk_reminder' | 'nearby_match' | 'event_suggestion' | 'milestone' | 'other';
  title: string;
  message: string;
  actionUrl?: string;
  actionLabel?: string;
  priority: 'low' | 'medium' | 'high';
  status: 'active' | 'dismissed' | 'accepted';
  expiresAt?: string;
  metadata?: Record<string, any>;
  createdAt: string;
}

export interface Match {
  id: string;
  userId: string;
  targetUserId: string;
  targetUser?: User;
  petId: string;
  pet?: Pet;
  targetPetId: string;
  targetPet?: Pet;
  score: number;
  compatibilityFactors?: {
    energyLevel: number;
    playStyle: number;
    size: number;
    location: number;
  };
  status: 'pending' | 'accepted' | 'declined';
  createdAt: string;
}

export interface ServiceProvider {
  id: string;
  name: string;
  type: 'trainer' | 'groomer' | 'vet' | 'daycare' | 'walker' | 'other';
  description: string;
  location: {
    latitude: number;
    longitude: number;
    address: string;
  };
  rating?: number;
  reviewsCount?: number;
  priceRange?: string;
  phone?: string;
  website?: string;
  imageUrl?: string;
  verified: boolean;
  createdAt: string;
}

export interface Notification {
  id: string;
  userId: string;
  type: 'like' | 'comment' | 'follow' | 'message' | 'event' | 'nudge' | 'system';
  title: string;
  message: string;
  actionUrl?: string;
  read: boolean;
  metadata?: Record<string, any>;
  createdAt: string;
}

export interface LeaderboardEntry {
  rank: number;
  userId: string;
  user?: User;
  points: number;
  level: number;
  badges: string[];
  activitiesCount: number;
  weeklyPoints?: number;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'activity' | 'social' | 'milestone' | 'special';
  requirement: string;
  unlockedAt?: string;
}

export interface CoActivity {
  id: string;
  activityId: string;
  activity?: Activity;
  participants: Array<{
    userId: string;
    user?: User;
    petId?: string;
    pet?: Pet;
  }>;
  status: 'active' | 'completed';
  startTime: string;
  endTime?: string;
  createdAt: string;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  message?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

// Form types
export interface CreatePetDto {
  name: string;
  species: 'dog' | 'cat' | 'other';
  breed?: string;
  age?: number;
  weight?: number;
  gender?: 'male' | 'female' | 'unknown';
  bio?: string;
  energyLevel?: number;
  playStyle?: string[];
  temperament?: string[];
}

export interface CreateActivityDto {
  type: 'walk' | 'run' | 'play' | 'training' | 'other';
  title: string;
  description?: string;
  duration: number;
  distance?: number;
  startTime: string;
  endTime?: string;
  petId: string;
  location?: {
    latitude: number;
    longitude: number;
    address?: string;
  };
}

export interface CreatePostDto {
  content: string;
  petId?: string;
  activityId?: string;
}

export interface CreateEventDto {
  title: string;
  description: string;
  startTime: string;
  endTime: string;
  location: {
    latitude: number;
    longitude: number;
    address: string;
  };
  capacity?: number;
}

export interface UpdateUserDto {
  displayName?: string;
  bio?: string;
  location?: string;
}
