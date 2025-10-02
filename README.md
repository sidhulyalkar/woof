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

## 🚀 Quick Start

### Prerequisites

- **Node.js** 20+
- **pnpm** 8+
- **Docker & Docker Compose**

### Setup in 4 Steps

```bash
# 1. Install dependencies
pnpm install

# 2. Start Docker services
docker compose up -d

# 3. Set up database
cp packages/database/.env.example packages/database/.env
pnpm --filter @woof/database db:generate
pnpm --filter @woof/database db:migrate
pnpm --filter @woof/database db:seed

# 4. Start API & Frontend
cp apps/api/.env.example apps/api/.env
cp apps/web/.env.local.example apps/web/.env.local

# Terminal 1: API
pnpm --filter @woof/api dev

# Terminal 2: Web
pnpm --filter @woof/web dev
```

**Frontend**: http://localhost:3000
**API**: http://localhost:4000
**Swagger Docs**: http://localhost:4000/docs

📖 **See [DEVELOPMENT.md](./DEVELOPMENT.md) for complete guide**

---

## 🏗️ Monorepo Structure

```
woof/
├── apps/
│   ├── api/          # NestJS backend ✅
│   ├── web/          # Next.js 15 frontend ✅
│   └── mobile/       # Expo React Native 📋
├── packages/
│   ├── ui/           # Galaxy-dark brand system ✅
│   ├── database/     # Prisma + PostgreSQL + pgvector ✅
│   └── config/       # Shared TypeScript/ESLint configs ✅
├── infra/
│   ├── db/           # Database scripts ✅
│   └── docker/       # Dockerfiles ✅
└── docker-compose.yml # PostgreSQL + Redis + n8n ✅
```

**Legend:** ✅ Complete | 📋 Planned

---

## 🎨 Galaxy-Dark Brand

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
```

**Asymmetric neuron logo** with leash curve → paw print (SVG included)

---

## 📦 Packages

### @woof/api ✅
NestJS backend with JWT auth, Swagger docs, 6 core modules (auth, users, pets, activities, social, meetups, compatibility).

### @woof/ui ✅
Galaxy-dark design system with 100+ color tokens, typography, spacing, asymmetric neuron logo, glassmorphism utilities.

### @woof/database ✅
Prisma schema with 15 models, pgvector for ML compatibility, comprehensive seed data (3 users, 3 pets, activities).

### @woof/config ✅
Shared TypeScript and ESLint configurations for Next.js, React Native, and Node.js.

---

## 🛠️ Development

```bash
# Install dependencies
pnpm install

# Start all apps
pnpm dev

# Database commands
pnpm db:migrate    # Run migrations
pnpm db:seed       # Seed demo data
pnpm db:studio     # Open Prisma Studio

# Build & test
pnpm build         # Build all packages
pnpm lint          # Lint all code
pnpm test          # Run tests
```

---

## 📚 Documentation

- **[DEVELOPMENT.md](./DEVELOPMENT.md)** - Complete development guide
- **[apps/api/README.md](./apps/api/README.md)** - API documentation
- **[docs/api-spec.md](./docs/api-spec.md)** - Legacy API spec (reference)

---

## 🗄️ Database Models

15 Prisma models with pgvector for ML compatibility:

- **User** - Profiles, auth, points
- **Pet** - Animals with vector embeddings
- **PetEdge** - Social graph with compatibility scores
- **Activity** - Walks, runs, plays with GeoJSON routes
- **Post, Like, Comment** - Social feed
- **Meetup, MeetupInvite** - Event coordination
- **And more...**

---

## 🚢 Deployment

- **Web**: Vercel
- **API**: Fly.io / Railway
- **Mobile**: Expo EAS
- **Database**: Neon / Supabase (PostgreSQL + pgvector)
- **Storage**: Cloudflare R2

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit with conventional commits
4. Push and open PR

**Standards**: Strict TypeScript, ESLint 0 errors, Prettier formatting

---

## 📄 License

MIT © [Sidharth Hulyalkar](https://github.com/sidhulyalkar)

---

**Woof** - Where technology meets companionship 🐾
