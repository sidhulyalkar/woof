import type {
  Owner,
  Pet,
  Match,
  Message,
  Event,
  Post,
  Badge,
  UserStats,
  LeaderboardEntry,
  Friend,
  FriendRequest,
  HealthRecord,
  MoodEntry,
  EnrichmentActivity,
  BehaviorLog,
  Highlight,
  MapMarker,
} from "./types"

// Mock Owners
export const mockOwners: Owner[] = [
  {
    id: "o1",
    name: "Sarah Chen",
    age: 28,
    location: {
      lat: 37.7749,
      lng: -122.4194,
      address: "San Francisco, CA",
    },
    bio: "Dog mom to Luna. Love hiking and beach days!",
    avatarUrl: "/user-avatar.jpg",
    preferences: {
      activityLevel: "high",
      schedule: ["morning", "evening"],
      interests: ["hiking", "beach", "training"],
    },
  },
  {
    id: "o2",
    name: "Mike Rodriguez",
    age: 32,
    location: {
      lat: 37.8044,
      lng: -122.2712,
      address: "Oakland, CA",
    },
    bio: "Husky dad. Always up for adventures!",
    avatarUrl: "/man-with-husky.jpg",
    preferences: {
      activityLevel: "high",
      schedule: ["morning", "afternoon"],
      interests: ["running", "hiking", "agility"],
    },
  },
]

// Mock Pets
export const mockPets: Pet[] = [
  {
    id: "p1",
    ownerId: "o1",
    name: "Luna",
    species: "dog",
    breed: "Border Collie",
    age: 3,
    size: "medium",
    temperament: ["friendly", "energetic", "intelligent"],
    photoUrl: "/border-collie.jpg",
  },
  {
    id: "p2",
    ownerId: "o2",
    name: "Max",
    species: "dog",
    breed: "Siberian Husky",
    age: 2,
    size: "large",
    temperament: ["playful", "energetic", "social"],
    photoUrl: "/siberian-husky-portrait.png",
  },
]

// Mock Matches
export const mockMatches: Match[] = [
  {
    id: "m1",
    owner: mockOwners[0],
    pet: mockPets[0],
    compatibility: {
      overall: 92,
      factors: {
        schedule: 95,
        location: 88,
        activityLevel: 90,
        petCompatibility: 94,
        interests: 92,
      },
      explanation: [
        "Both prefer morning walks",
        "Live within 2 miles",
        "Similar activity levels",
        "Pets have compatible temperaments",
      ],
    },
    distance: 1.2,
    matchedAt: new Date().toISOString(),
  },
]

// Mock Messages
export const mockMessages: Message[] = [
  {
    id: "msg1",
    conversationId: "conv1",
    senderId: "o1",
    content: "Hey! Would love to set up a playdate for our dogs!",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    read: true,
    type: "text",
  },
  {
    id: "msg2",
    conversationId: "conv1",
    senderId: "o2",
    content: "That sounds great! How about this Saturday at the dog park?",
    timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
    read: false,
    type: "text",
  },
]

// Mock Events
export const mockEvents: Event[] = [
  {
    id: "e1",
    title: "Dog Park Meetup",
    description: "Weekly meetup for dogs to socialize and play",
    location: {
      lat: 37.7699,
      lng: -122.4194,
      address: "Golden Gate Park, San Francisco",
    },
    datetime: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
    duration: 120,
    capacity: 20,
    attendees: ["o1", "o2"],
    organizerId: "o1",
    category: "playdate",
    imageUrl: "/golden-gate-park.jpg",
  },
]

// Mock Posts
export const mockPosts: Post[] = [
  {
    id: "post1",
    userId: "o1",
    userName: "Sarah Chen",
    userAvatar: "/user-avatar.jpg",
    petName: "Luna",
    petAvatar: "/border-collie.jpg",
    mediaUrl: "/golden-retriever.png",
    mediaType: "image",
    caption: "Beautiful morning walk at the park!",
    location: "Golden Gate Park",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    likes: 42,
    comments: 8,
    isLiked: false,
  },
]

// Mock Badges
export const mockBadges: Badge[] = [
  {
    id: "b1",
    name: "Social Butterfly",
    description: "Made 10 new friends",
    iconUrl: "/badge-social.png",
    category: "social",
    rarity: "common",
    unlockedAt: new Date().toISOString(),
  },
  {
    id: "b2",
    name: "Marathon Walker",
    description: "Walked 100 miles total",
    iconUrl: "/badge-walker.png",
    category: "activity",
    rarity: "rare",
    unlockedAt: new Date().toISOString(),
  },
]

// Mock User Stats
export const mockUserStats: UserStats = {
  userId: "o1",
  points: 2450,
  level: 12,
  rank: 42,
  badges: mockBadges,
  streaks: {
    daily: 7,
    weekly: 3,
  },
  achievements: {
    totalWalks: 156,
    totalDistance: 234.5,
    totalEvents: 23,
    totalFriends: 18,
    totalPosts: 45,
  },
}

// Mock Leaderboard
export const mockLeaderboard: LeaderboardEntry[] = [
  {
    rank: 1,
    userId: "o1",
    userName: "Sarah Chen",
    userAvatar: "/user-avatar.jpg",
    petName: "Luna",
    petAvatar: "/border-collie.jpg",
    points: 3250,
    level: 15,
    badges: 12,
    change: 2,
  },
  {
    rank: 2,
    userId: "o2",
    userName: "Mike Rodriguez",
    userAvatar: "/man-with-husky.jpg",
    petName: "Max",
    petAvatar: "/siberian-husky-portrait.png",
    points: 3100,
    level: 14,
    badges: 10,
    change: -1,
  },
]

// Mock Friends
export const mockFriends: Friend[] = [
  {
    id: "o1",
    name: "Sarah Chen",
    avatarUrl: "/user-avatar.jpg",
    petName: "Luna",
    petAvatar: "/border-collie.jpg",
    location: "San Francisco, CA",
    mutualFriends: 5,
    status: "friends",
    friendsSince: "2024-01-15",
  },
]

// Mock Friend Requests
export const mockFriendRequests: FriendRequest[] = [
  {
    id: "fr1",
    fromUserId: "o2",
    fromUserName: "Mike Rodriguez",
    fromUserAvatar: "/man-with-husky.jpg",
    fromPetName: "Max",
    fromPetAvatar: "/siberian-husky-portrait.png",
    message: "Hey! Our dogs would love to play together!",
    timestamp: new Date().toISOString(),
  },
]

// Mock Health Records
export const mockHealthRecords: HealthRecord[] = [
  {
    id: "h1",
    petId: "p1",
    type: "vet-visit",
    date: "2024-03-10",
    title: "Annual Checkup",
    description: "Routine examination, all vitals normal",
    veterinarian: "Dr. Sarah Johnson",
    nextDue: "2025-03-10",
  },
  {
    id: "h2",
    petId: "p1",
    type: "vaccination",
    date: "2024-03-10",
    title: "Rabies Vaccine",
    veterinarian: "Dr. Sarah Johnson",
    nextDue: "2027-03-10",
  },
]

// Mock Mood Entries
export const mockMoodEntries: MoodEntry[] = [
  {
    id: "mood1",
    petId: "p1",
    date: "2024-03-15",
    mood: "happy",
    notes: "Great energy after morning walk",
    activities: ["Walk", "Play fetch"],
  },
]

// Mock Enrichment Activities
export const mockEnrichmentActivities: EnrichmentActivity[] = [
  {
    id: "enrich1",
    name: "Puzzle Feeder",
    category: "mental",
    description: "Hide treats in a puzzle toy to stimulate problem-solving",
    duration: 15,
    difficulty: "medium",
  },
]

// Mock Behavior Logs
export const mockBehaviorLogs: BehaviorLog[] = [
  {
    id: "beh1",
    petId: "p1",
    date: "2024-03-15",
    behavior: "Learned new trick",
    severity: "positive",
    context: "Training session",
    notes: "Successfully learned 'spin' command",
  },
]

// Mock Highlights
export const mockHighlights: Highlight[] = [
  {
    id: "hl1",
    userId: "o1",
    userName: "Sarah Chen",
    userAvatar: "/user-avatar.jpg",
    petName: "Luna",
    petAvatar: "/border-collie.jpg",
    videoUrl: "/dog-fetching-ball.png",
    thumbnail: "/border-collie.jpg",
    caption: "Luna's first time catching a frisbee!",
    timestamp: new Date().toISOString(),
    views: 234,
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
  },
]

// Mock Map Markers
export const mockMapMarkers: MapMarker[] = [
  {
    id: "marker1",
    type: "pet",
    lat: 37.7749,
    lng: -122.4194,
    title: "Luna",
    subtitle: "with Sarah Chen",
    avatarUrl: "/border-collie.jpg",
    data: { distance: 0.3, compatibility: 92 },
  },
  {
    id: "marker2",
    type: "event",
    lat: 37.7699,
    lng: -122.4194,
    title: "Dog Park Meetup",
    subtitle: "Today at 3:00 PM",
    data: { attendees: 12, capacity: 20 },
  },
]
