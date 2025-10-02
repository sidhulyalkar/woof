# Woof Web Application

Next.js 15 frontend for the Woof pet social fitness platform.

## Features

- **Galaxy-Dark Theme**: Custom dark theme with neuron-inspired design
- **Type-Safe API Integration**: Full TypeScript integration with NestJS backend
- **Real-time Updates**: React Query for optimistic updates and caching
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Component Library**: Radix UI primitives with custom styling

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **State Management**: Zustand + React Query
- **UI Components**: Radix UI + Custom Components
- **API Client**: Axios with interceptors
- **Fonts**: Space Grotesk (headings), Inter (body)

## Getting Started

### Prerequisites

- Node.js 20+
- pnpm 9+
- API server running on `http://localhost:4000`

### Installation

From the monorepo root:

```bash
pnpm install
```

### Development

```bash
# Start dev server
pnpm --filter @woof/web dev

# Type check
pnpm --filter @woof/web type-check

# Lint
pnpm --filter @woof/web lint

# Build for production
pnpm --filter @woof/web build
```

The app will be available at [http://localhost:3000](http://localhost:3000)

## Project Structure

```
apps/web/
├── src/
│   ├── app/                    # Next.js app router pages
│   │   ├── page.tsx           # Dashboard
│   │   ├── pets/              # Pet management
│   │   ├── activities/        # Activity tracking
│   │   ├── social/            # Social feed
│   │   ├── leaderboard/       # Leaderboards
│   │   └── layout.tsx         # Root layout
│   ├── components/
│   │   ├── layout/            # Header, navigation
│   │   ├── dashboard/         # Dashboard components
│   │   ├── social/            # Social feed components
│   │   └── ui/                # Reusable UI components
│   ├── lib/
│   │   ├── api.ts             # API client & endpoints
│   │   └── utils.ts           # Utility functions
│   └── hooks/                 # Custom React hooks
├── public/                     # Static assets
└── package.json
```

## API Integration

The app connects to the NestJS API via axios. All API calls are in `src/lib/api.ts`:

- **Auth**: Login, register, profile
- **Pets**: CRUD operations
- **Activities**: Track and view activities
- **Social**: Feed, posts, likes, comments
- **Leaderboard**: Global and local rankings

### Authentication

JWT tokens are stored in localStorage and automatically attached to requests via axios interceptors.

## Environment Variables

Create `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:4000/api/v1
```

## Theme Customization

The galaxy-dark theme is defined in:

- `tailwind.config.ts` - Color palette and design tokens
- `src/app/globals.css` - CSS variables and utilities

### Colors

- **Primary**: `#0B1C3D` - Deep space blue
- **Secondary**: `#0F2A6B` - Midnight blue
- **Accent**: `#6BA8FF` - Bright electric blue
- **Surface**: `#0E1220` - Dark background

## Components

### Layout Components

- `Header` - Main navigation with logo and user menu
- `MobileNav` - Bottom navigation for mobile
- `UserMenu` - User profile dropdown

### Dashboard Components

- `PetCard` - Display pet info and daily stats
- `HappinessScore` - Circular progress indicator
- `WeeklyProgress` - Activity progress tracking

### Social Components

- `SocialFeed` - Post feed with likes/comments
- `HappinessMetrics` - Weekly happiness tracking

## Development Guidelines

### Adding a New Page

1. Create page in `src/app/[page-name]/page.tsx`
2. Add route to navigation in `src/components/layout/header.tsx`
3. Create API functions in `src/lib/api.ts` if needed
4. Use React Query for data fetching

### Creating Components

- Use TypeScript for all components
- Follow the component structure in existing files
- Use Tailwind classes with the `cn()` utility
- Export components from component files

### API Calls

```typescript
import { useQuery, useMutation } from '@tanstack/react-query';
import { petsApi } from '@/lib/api';

// Fetch data
const { data, isLoading } = useQuery({
  queryKey: ['pets'],
  queryFn: petsApi.getAllPets,
});

// Mutate data
const mutation = useMutation({
  mutationFn: petsApi.createPet,
  onSuccess: () => {
    // Invalidate and refetch
  },
});
```

## Building for Production

```bash
# Build
pnpm --filter @woof/web build

# Start production server
pnpm --filter @woof/web start
```

## Troubleshooting

### Port already in use

Change the port in `package.json`:

```json
"dev": "next dev -p 3001"
```

### API connection errors

1. Ensure the API server is running on port 4000
2. Check `.env.local` has the correct API URL
3. Verify CORS is enabled on the API

### Type errors

Run type check to see all errors:

```bash
pnpm --filter @woof/web type-check
```

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com)
- [React Query](https://tanstack.com/query)
- [Radix UI](https://radix-ui.com)
