# 🐾 Woof - Pet Social Fitness Platform

> **Production-grade monorepo** for the pet-first social network with mutual fitness tracking, AI-powered meetups, and galaxy-dark aesthetic.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![pnpm](https://img.shields.io/badge/pnpm-8.15-orange)](https://pnpm.io/)
[![Turborepo](https://img.shields.io/badge/Turborepo-1.12-red)](https://turbo.build/)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

---

## 🎯 What is Woof?

Woof is a **Strava-grade, pet-first social platform** that unifies:

1. **Pet-Centric Social Graph** - Friendships, compatibility, interactions
2. **Mutual Fitness Tracking** - Human + pet activity sync (HealthKit/Google Fit)
3. **Proactive Meetups** - AI-powered location-based suggestions
4. **Gamification** - Points, achievements, leaderboards, rewards
5. **Premium UX** - Galaxy-dark aesthetic with glassmorphism

---

## 🏗️ Monorepo Structure

```
woof/
├── apps/
│   ├── web/          # Next.js 15 (App Router, Tailwind, shadcn/ui)
│   ├── mobile/       # Expo React Native (HealthKit/Google Fit)
│   ├── api/          # NestJS (Prisma, Socket.io, BullMQ)
│   └── automations/  # n8n workflows (Docker)
├── packages/
│   ├── ui/           # Brand system + shared components
│   ├── database/     # Prisma schema + client
│   ├── config/       # Shared configs (TS, ESLint)
│   └── sdk/          # Generated OpenAPI SDK
├── infra/
│   ├── docker/       # Docker configs
│   ├── db/           # Database scripts
│   └── ci/           # GitHub Actions
└── docs/
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 20+ ([Download](https://nodejs.org))
- **pnpm** 8+ (installed via setup script)
- **PostgreSQL** 15+ with pgvector
- **Docker** (optional, recommended)

### 1. Clone & Setup

```bash
git clone https://github.com/sidhulyalkar/woof.git
cd woof

# Run setup script
./scripts/setup.sh
```

### 2. Configure Database

```bash
# Update database credentials
nano packages/database/.env

# Example:
# DATABASE_URL="postgresql://postgres:password@localhost:5432/woof"
```

### 3. Initialize Database

```bash
# Generate Prisma client
pnpm --filter @woof/database db:generate

# Run migrations
pnpm --filter @woof/database db:migrate

# Seed demo data (3 users, 3 pets, sample activities)
pnpm --filter @woof/database db:seed
```

### 4. Start Development

```bash
# Start all apps
pnpm dev

# Or start specific apps
pnpm --filter web dev
pnpm --filter api dev
```

---

## 🎨 Brand System

### Galaxy-Dark Theme

```typescript
import { colors, fonts, theme } from '@woof/ui/theme';

// Brand Colors
colors.primary    // #0B1C3D - Deep cosmic blue
colors.secondary  // #0F2A6B - Rich navy
colors.accent     // #6BA8FF - Bright blue
colors.surface    // #0E1220 - Dark slate

// Typography
fonts.heading     // Space Grotesk
fonts.body        // Inter
fonts.mono        // Fira Code

// Glassmorphism
theme.glassmorphism.background       // rgba(14, 18, 32, 0.7)
theme.glassmorphism.backdropFilter   // blur(16px)
```

### Logo

Asymmetric neuron silhouette forming a leash curve around a paw.

```typescript
import Logo from '@woof/ui/theme/logo.svg';
```

---

## 📦 Packages

### @woof/ui

Shared component library with galaxy-dark theme.

```bash
pnpm --filter @woof/ui build
```

**Exports:**
- Brand colors, typography, spacing
- Asymmetric neuron logo (SVG)
- Glassmorphism utilities
- Motion/spring presets

### @woof/database

Prisma schema with PostgreSQL + pgvector support.

```bash
# Generate Prisma client
pnpm --filter @woof/database db:generate

# Run migrations
pnpm --filter @woof/database db:migrate

# Open Prisma Studio
pnpm --filter @woof/database db:studio

# Seed database
pnpm --filter @woof/database db:seed
```

**Entities:**
- User, Pet, Device
- Activity, MutualGoal, Reward
- PetEdge (social graph with pgvector)
- Meetup, MeetupInvite
- Post, Like, Comment
- Notification, IntegrationToken
- Place, Telemetry

### @woof/config

Shared TypeScript and ESLint configurations.

```json
{
  "extends": "@woof/config/typescript/nextjs"
}
```

---

## 🛠️ Development

### Available Scripts

```bash
# Development
pnpm dev                    # Start all apps
pnpm build                  # Build all apps
pnpm lint                   # Lint all packages
pnpm test                   # Run all tests
pnpm clean                  # Clean all builds

# Database
pnpm db:migrate             # Run migrations
pnpm db:seed                # Seed demo data
pnpm db:studio              # Open Prisma Studio

# Utilities
pnpm format                 # Format code with Prettier
pnpm artifact               # Create release ZIP
```

### Project Commands

```bash
# Work with specific packages
pnpm --filter web build
pnpm --filter api dev
pnpm --filter @woof/database db:studio

# Add dependencies
pnpm --filter web add axios
pnpm --filter api add @nestjs/jwt

# Run commands in all packages
pnpm -r build
```

---

## 🗄️ Database Schema

### Key Models

**User**
- handle, email, bio, location
- points, visibility settings
- Relations: pets, activities, posts

**Pet**
- name, species, breed, temperament
- embedding (pgvector for ML)
- Relations: owner, activities, friends

**PetEdge**
- Social graph connections
- Compatibility scores
- Interaction history

**Activity**
- Type (WALK, RUN, PLAY, HIKE)
- GeoJSON routes
- Human + Pet metrics

**Post**
- Social feed content
- Media attachments
- Visibility controls

### pgvector Integration

```typescript
// Pet compatibility uses vector embeddings
model Pet {
  embedding Unsupported("vector(384)")?
}

// Query similar pets
SELECT * FROM pets
ORDER BY embedding <-> $1
LIMIT 10;
```

---

## 🧪 Testing

```bash
# Run all tests
pnpm test

# Run specific package tests
pnpm --filter api test
pnpm --filter web test
```

**Test Stack:**
- Vitest (unit tests)
- Playwright (E2E)
- Supertest (API tests)
- Detox (mobile tests)

---

## 🐳 Docker

### Development

```bash
# Start all services
docker compose up -d

# Services available:
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
# - n8n: localhost:5678
```

### Production

```bash
# Build images
docker build -f infra/docker/Dockerfile.web -t woof-web .
docker build -f infra/docker/Dockerfile.api -t woof-api .
```

---

## 📚 Documentation

- [Migration Plan](./MIGRATION_PLAN.md) - Hybrid migration strategy
- [Progress Tracker](./PROGRESS.md) - Current development status
- [API Spec](./docs/api-spec.md) - REST API documentation
- [Data Models](./docs/data-models.md) - Database schemas

---

## 🚢 Deployment

### Vercel (Web)
```bash
cd apps/web
vercel
```

### Fly.io (API)
```bash
cd apps/api
fly deploy
```

### Expo EAS (Mobile)
```bash
cd apps/mobile
eas build --platform all
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Code Standards:**
- Strict TypeScript
- ESLint 0 errors
- Prettier formatting
- Conventional commits

---

## 📄 License

MIT © [Sidharth Hulyalkar](https://github.com/sidhulyalkar)

---

## 🙏 Acknowledgments

- **Brand**: Galaxy-dark aesthetic inspired by modern design systems
- **Architecture**: NestJS + Prisma + Expo best practices
- **Community**: Built with ❤️ for pet lovers

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sidhulyalkar/woof/issues)
- **Docs**: See `/docs` directory
- **Email**: sidharth@woof.com

---

**Woof** - Where technology meets companionship 🐾
