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

- **Node.js** 20+ ([Download](https://nodejs.org))
- **pnpm** 8+ (auto-installed via setup script)
- **PostgreSQL** 15+ with pgvector
- **Docker** (optional, recommended)

### Setup in 3 Steps

```bash
# 1. Run setup script
./scripts/setup.sh

# 2. Configure database
cp packages/database/.env.example packages/database/.env
# Edit packages/database/.env with your PostgreSQL URL

# 3. Initialize database
pnpm --filter @woof/database db:generate
pnpm --filter @woof/database db:migrate
pnpm --filter @woof/database db:seed
```

📖 **See [QUICK_START.md](./QUICK_START.md) for detailed setup guide**

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
│   ├── ui/           # Brand system + shared components ✅
│   ├── database/     # Prisma schema + client ✅
│   ├── config/       # Shared configs (TS, ESLint) ✅
│   └── sdk/          # Generated OpenAPI SDK
└── infra/            # Docker, CI/CD
```

**Legend:** ✅ Complete | ⏳ In Progress | 📋 Planned

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

### @woof/ui ✅
Galaxy-dark design system with 100+ color tokens, typography, spacing, logo, and glassmorphism utilities.

### @woof/database ✅
Prisma schema with 15 models (User, Pet, Activity, Post, etc.), pgvector support for ML, comprehensive seed data.

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

- **[QUICK_START.md](./QUICK_START.md)** - 5-minute setup guide
- **[MIGRATION_PLAN.md](./MIGRATION_PLAN.md)** - Development strategy
- **[PROGRESS.md](./PROGRESS.md)** - Current status
- **[ACCOMPLISHMENTS.md](./ACCOMPLISHMENTS.md)** - What we've built
- **[docs/api-spec.md](./docs/api-spec.md)** - API documentation

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
