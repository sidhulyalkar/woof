# PetPath - Pet Owner Social Network

A comprehensive social networking platform for pet owners to connect, share experiences, and manage their pets' health and activities.

## Features

### Core Features
- **Smart Matching**: AI-powered compatibility matching based on location, schedule, activity level, and pet temperament
- **Social Feed**: Share photos, videos, and updates about your pets
- **Messaging**: Real-time chat with other pet owners
- **Events**: Create and join local pet-friendly events and playdates
- **Profile Management**: Detailed profiles for both owners and pets

### Advanced Features
- **Gamification**: Points, levels, badges, and leaderboards to encourage engagement
- **Friends Management**: Connect with other pet owners, send/accept friend requests
- **Health Tracking**: Vet visits, vaccinations, medications, and weight monitoring
- **Wellness**: Mood tracking, enrichment activities, and behavior logs
- **Map & Highlights**: Discover nearby pets, events, and services; share short video stories
- **PWA Support**: Install as a native app with offline functionality

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui
- **State Management**: Zustand
- **Icons**: Lucide React
- **PWA**: Service Worker with offline support

## Getting Started

### Installation

\`\`\`bash
# Install dependencies
npm install

# Run development server
npm run dev
\`\`\`

Open [http://localhost:3000](http://localhost:3000) to view the app.

### Project Structure

\`\`\`
app/
├── api/              # API routes (mock endpoints)
├── events/           # Events page
├── friends/          # Friends management
├── health/           # Health tracking
├── highlights/       # Video stories
├── map/              # Map view
├── matches/          # Smart matching
├── messages/         # Messaging
├── profile/          # User profiles
├── settings/         # App settings
├── wellness/         # Mental wellness
└── page.tsx          # Home feed

components/
├── bottom-nav.tsx    # Bottom navigation
├── feed/             # Feed components
├── highlights/       # Highlights components
├── matches/          # Match components
└── ui/               # shadcn/ui components

lib/
├── types.ts          # TypeScript types
├── mock-data.ts      # Mock data for testing
├── api-client.ts     # API client utilities
└── stores/           # Zustand stores
\`\`\`

## API Endpoints

All API endpoints are mocked for development:

- `GET /api/matches` - Get compatible matches
- `GET /api/messages` - Get messages
- `POST /api/messages` - Send message
- `GET /api/events` - Get events
- `POST /api/events` - Create event
- `GET /api/posts` - Get feed posts
- `POST /api/posts` - Create post
- `GET /api/stats` - Get user stats
- `GET /api/leaderboard` - Get leaderboard
- `GET /api/friends` - Get friends
- `POST /api/friends` - Send friend request
- `GET /api/health` - Get health records
- `POST /api/health` - Add health record

## PWA Features

- **Installable**: Add to home screen on mobile devices
- **Offline Support**: Service worker caches essential resources
- **App Icons**: Custom icons for iOS and Android
- **Manifest**: Full PWA manifest configuration

## Development

### Mock Data

All data is currently mocked in `lib/mock-data.ts`. To connect to a real backend:

1. Update API endpoints in `lib/api-client.ts`
2. Replace mock data with real API calls
3. Add authentication and authorization

### Testing

Test utilities are available in `lib/test-utils.tsx` for component testing.

## Deployment

Deploy to Vercel with one click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

## License

MIT
