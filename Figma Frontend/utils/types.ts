// User and Profile Types
export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface Profile {
  id: string;
  name: string;
  email: string;
  pet_name: string;
  pet_type: string;
  avatar: string;
  pet_avatar?: string;
  bio: string;
  total_activities: number;
  total_distance: number;
  total_points: number;
  created_at: string;
  updated_at: string;
}

// Post Types
export interface Post {
  id: string;
  user_id: string;
  content: string;
  image_url?: string;
  likes_count: number;
  comments_count: number;
  activity_data?: ActivityData;
  created_at: string;
  user?: {
    name: string;
    avatar: string;
    pet_name: string;
    pet_type: string;
    pet_avatar?: string;
  };
}

export interface ActivityData {
  type: string;
  distance?: string;
  duration?: string;
  course?: string;
  score?: string;
  calories?: number;
}

// Activity Types
export interface Activity {
  id: string;
  user_id: string;
  type: string;
  distance?: number;
  duration: number;
  calories?: number;
  points: number;
  notes?: string;
  score?: number;
  created_at: string;
}

export type ActivityType = 
  | 'running' 
  | 'walking' 
  | 'hiking' 
  | 'swimming' 
  | 'agility' 
  | 'training' 
  | 'playing' 
  | 'stretching'
  | 'other';

// Leaderboard Types
export interface LeaderboardEntry {
  user_id: string;
  name: string;
  pet_name: string;
  pet_type: string;
  avatar: string;
  pet_avatar?: string;
  points: number;
  activities: number;
  distance: number;
}

export type LeaderboardTimeframe = 'weekly' | 'monthly' | 'yearly';

// Message Types
export interface Message {
  id: string;
  sender_id: string;
  recipient_id: string;
  content: string;
  type: 'text' | 'image' | 'activity';
  read: boolean;
  created_at: string;
}

export interface Conversation {
  id: string;
  participants: Profile[];
  last_message: Message;
  unread_count: number;
  updated_at: string;
}

// Auth Types
export interface AuthResponse {
  success: boolean;
  user?: User;
  profile?: Profile;
  session?: any;
  error?: string;
}

export interface SignupData {
  email: string;
  password: string;
  name: string;
  petName: string;
  petType: string;
}

export interface SigninData {
  email: string;
  password: string;
}

// API Response Types
export interface ApiResponse<T = any> {
  success?: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface FeedResponse extends ApiResponse {
  posts: Post[];
}

export interface ActivitiesResponse extends ApiResponse {
  activities: Activity[];
}

export interface LeaderboardResponse extends ApiResponse {
  leaderboard: LeaderboardEntry[];
}

export interface MessagesResponse extends ApiResponse {
  messages: Message[];
}

export interface ProfileResponse extends ApiResponse {
  profile: Profile;
  activities: Activity[];
  posts: Post[];
}