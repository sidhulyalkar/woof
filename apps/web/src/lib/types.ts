// Core client-side product types for Woof.

export interface Owner {
  id: string;
  name: string;
  bio?: string;
  avatarUrl?: string;
  isVerified?: boolean;
  age?: number;
  location?: string | { lat: number; lng: number; address?: string };
}

export interface Pet {
  id: string;
  ownerId: string;
  name: string;
  species: string;
  breed?: string;
  age?: number;
  size?: string;
  temperament: string[];
  photoUrl?: string;
}

export interface CompatibilityScore {
  overall: number;
  confidence: number;
  source: string;
  factors: {
    species: number;
    temperament?: number;
    age?: number;
    breed?: number;
    schedule?: number;
  };
  explanation: string[];
}

export interface Match {
  id: string;
  owner: Owner;
  pet: Pet;
  compatibility: CompatibilityScore;
  status?: 'PROPOSED' | 'CONFIRMED' | 'AVOID';
  matchedAt?: string;
  reasons?: string[];
  commonInterests?: string[];
}

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  content: string;
  timestamp: string;
  read: boolean;
  type: 'text' | 'meetup-proposal';
  metadata?: {
    location?: { lat: number; lng: number; name: string };
    datetime?: string;
  };
}

export interface Event {
  id: string;
  title: string;
  description: string;
  location: {
    lat: number;
    lng: number;
    address: string;
  };
  datetime: string;
  duration: number;
  capacity: number;
  attendees: string[];
  organizerId: string;
  category: 'playdate' | 'training' | 'social' | 'other';
  imageUrl?: string;
}

export type ActivityType = 'walk' | 'play' | 'playdate' | 'training' | 'vet' | 'other';

export interface Activity {
  id: string;
  petId: string;
  type: ActivityType;
  startTime: string;
  endTime: string;
  duration: number;
  distance?: number;
  route?: Array<{ lat: number; lng: number }>;
  location?: { lat: number; lng: number };
  participants?: string[];
  notes?: string;
}

export interface Post {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  petName: string;
  petAvatar: string;
  mediaUrl: string;
  mediaType: 'image' | 'video';
  caption: string;
  location?: string;
  timestamp: string;
  likes: number;
  comments: number;
  isLiked: boolean;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  iconUrl: string;
  category: 'social' | 'activity' | 'health' | 'special';
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  unlockedAt?: string;
}

export interface UserStats {
  userId: string;
  points: number;
  level: number;
  rank: number;
  badges: Badge[];
  streaks: {
    daily: number;
    weekly: number;
  };
  achievements: {
    totalWalks: number;
    totalDistance: number;
    totalEvents: number;
    totalFriends: number;
    totalPosts: number;
  };
}

export interface LeaderboardEntry {
  rank: number;
  userId: string;
  userName: string;
  userAvatar: string;
  petName: string;
  petAvatar: string;
  points: number;
  level: number;
  badges: number;
  change: number;
}

export interface Friend {
  id: string;
  name: string;
  avatarUrl: string;
  petName: string;
  petAvatar: string;
  location: string;
  mutualFriends: number;
  status: 'friends' | 'pending' | 'requested' | 'none';
  friendsSince?: string;
}

export interface FriendRequest {
  id: string;
  fromUserId: string;
  fromUserName: string;
  fromUserAvatar: string;
  fromPetName: string;
  fromPetAvatar: string;
  message?: string;
  timestamp: string;
}

export interface HealthRecord {
  id: string;
  petId: string;
  type: 'vet-visit' | 'vaccination' | 'medication' | 'weight';
  date: string;
  title: string;
  description?: string;
  veterinarian?: string;
  nextDue?: string;
  metadata?: {
    weight?: number;
    medication?: string;
    dosage?: string;
    frequency?: string;
  };
}

export interface MoodEntry {
  id: string;
  petId: string;
  date: string;
  mood: 'happy' | 'calm' | 'anxious' | 'energetic' | 'tired';
  notes?: string;
  activities: string[];
}

export interface EnrichmentActivity {
  id: string;
  name: string;
  category: 'mental' | 'physical' | 'social';
  description: string;
  duration: number;
  difficulty: 'easy' | 'medium' | 'hard';
}

export interface BehaviorLog {
  id: string;
  petId: string;
  date: string;
  behavior: string;
  severity: 'positive' | 'neutral' | 'concerning';
  context?: string;
  notes?: string;
}

export interface Highlight {
  id: string;
  userId: string;
  userName: string;
  userAvatar: string;
  petName: string;
  petAvatar: string;
  videoUrl: string;
  thumbnail: string;
  caption?: string;
  timestamp: string;
  views: number;
  expiresAt: string;
}

type MapMarkerBase = {
  id: string;
  lat: number;
  lng: number;
  title: string;
  subtitle?: string;
  avatarUrl?: string;
};

export type MapMarker =
  | (MapMarkerBase & {
      type: 'pet';
      data: { distance: number; compatibility: number };
    })
  | (MapMarkerBase & {
      type: 'event';
      data: { attendees: number; capacity: number };
    })
  | (MapMarkerBase & {
      type: 'service';
      data: { rating: number; reviews: number };
    });

export interface Service {
  id: string;
  name: string;
  category:
    'walker' | 'grooming' | 'vet' | 'sitter' | 'food-store' | 'restaurant' | 'park' | 'hike';
  location: {
    lat: number;
    lng: number;
    address: string;
  };
  rating: number;
  reviews: number;
  distance: number;
  priceRange?: '$' | '$$' | '$$$';
  hours?: string;
  phone?: string;
  imageUrl?: string;
  description?: string;
}

export interface ServiceProvider {
  id: string;
  userId: string;
  name: string;
  avatarUrl?: string;
  serviceType: 'dog-walking' | 'pet-sitting' | 'training' | 'grooming';
  bio?: string;
  rating: number;
  reviewCount: number;
  priceRange: string;
  location: { lat: number; lng: number; address: string };
  distance: number;
  availability: string[];
  verified: boolean;
}
