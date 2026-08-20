// Core data types for PetPath app

export interface Owner {
  id: string
  name: string
  age: number
  location: {
    lat: number
    lng: number
    address: string
  }
  bio: string
  avatarUrl: string
  preferences: {
    activityLevel: "low" | "medium" | "high"
    schedule: string[]
    interests: string[]
  }
}

export interface Pet {
  id: string
  ownerId: string
  name: string
  species: "dog" | "cat" | "other"
  breed: string
  age: number
  size: "small" | "medium" | "large"
  temperament: string[]
  photoUrl: string
  medicalNotes?: string
}

export interface CompatibilityScore {
  overall: number
  factors: {
    schedule: number
    location: number
    activityLevel: number
    petCompatibility: number
    interests: number
  }
  explanation: string[]
}

export interface Match {
  id: string
  owner: Owner
  pet: Pet
  compatibility: CompatibilityScore
  distance: number
  matchedAt: string
}

export interface Message {
  id: string
  conversationId: string
  senderId: string
  content: string
  timestamp: string
  read: boolean
  type: "text" | "meetup-proposal"
  metadata?: {
    location?: { lat: number; lng: number; name: string }
    datetime?: string
  }
}

export interface Event {
  id: string
  title: string
  description: string
  location: {
    lat: number
    lng: number
    address: string
  }
  datetime: string
  duration: number
  capacity: number
  attendees: string[]
  organizerId: string
  category: "playdate" | "training" | "social" | "other"
  imageUrl?: string
}

export interface Activity {
  id: string
  petId: string
  type: "walk" | "play" | "training" | "vet"
  startTime: string
  endTime: string
  distance?: number
  location?: { lat: number; lng: number }
  notes?: string
}

export interface Post {
  id: string
  userId: string
  userName: string
  userAvatar: string
  petName: string
  petAvatar: string
  mediaUrl: string
  mediaType: "image" | "video"
  caption: string
  location?: string
  timestamp: string
  likes: number
  comments: number
  isLiked: boolean
}

export interface Badge {
  id: string
  name: string
  description: string
  iconUrl: string
  category: "social" | "activity" | "health" | "special"
  rarity: "common" | "rare" | "epic" | "legendary"
  unlockedAt?: string
}

export interface UserStats {
  userId: string
  points: number
  level: number
  rank: number
  badges: Badge[]
  streaks: {
    daily: number
    weekly: number
  }
  achievements: {
    totalWalks: number
    totalDistance: number
    totalEvents: number
    totalFriends: number
    totalPosts: number
  }
}

export interface LeaderboardEntry {
  rank: number
  userId: string
  userName: string
  userAvatar: string
  petName: string
  petAvatar: string
  points: number
  level: number
  badges: number
  change: number // rank change from previous period
}

export interface Friend {
  id: string
  name: string
  avatarUrl: string
  petName: string
  petAvatar: string
  location: string
  mutualFriends: number
  status: "friends" | "pending" | "requested" | "none"
  friendsSince?: string
}

export interface FriendRequest {
  id: string
  fromUserId: string
  fromUserName: string
  fromUserAvatar: string
  fromPetName: string
  fromPetAvatar: string
  message?: string
  timestamp: string
}

export interface HealthRecord {
  id: string
  petId: string
  type: "vet-visit" | "vaccination" | "medication" | "weight"
  date: string
  title: string
  description?: string
  veterinarian?: string
  nextDue?: string
  metadata?: {
    weight?: number
    medication?: string
    dosage?: string
    frequency?: string
  }
}

export interface MoodEntry {
  id: string
  petId: string
  date: string
  mood: "happy" | "calm" | "anxious" | "energetic" | "tired"
  notes?: string
  activities: string[]
}

export interface EnrichmentActivity {
  id: string
  name: string
  category: "mental" | "physical" | "social"
  description: string
  duration: number
  difficulty: "easy" | "medium" | "hard"
}

export interface BehaviorLog {
  id: string
  petId: string
  date: string
  behavior: string
  severity: "positive" | "neutral" | "concerning"
  context?: string
  notes?: string
}

export interface Highlight {
  id: string
  userId: string
  userName: string
  userAvatar: string
  petName: string
  petAvatar: string
  videoUrl: string
  thumbnail: string
  caption?: string
  timestamp: string
  views: number
  expiresAt: string
}

export interface MapMarker {
  id: string
  type: "pet" | "event" | "service"
  lat: number
  lng: number
  title: string
  subtitle?: string
  avatarUrl?: string
  data: any
}

export interface Service {
  id: string
  name: string
  category: "walker" | "grooming" | "vet" | "sitter" | "food-store" | "restaurant" | "park" | "hike"
  location: {
    lat: number
    lng: number
    address: string
  }
  rating: number
  reviews: number
  distance: number
  priceRange?: "$" | "$$" | "$$$"
  hours?: string
  phone?: string
  imageUrl?: string
  description?: string
}
