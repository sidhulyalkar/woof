# 🚀 Woof - Quick Start Guide

**Get up and running in 5 minutes!**

---

## ⚡ Fast Track Setup

### 1. Install pnpm (if needed)
```bash
npm install -g pnpm
```

### 2. Install Dependencies
```bash
pnpm install
```

### 3. Set up Database
```bash
# Copy environment file
cp packages/database/.env.example packages/database/.env

# Edit with your PostgreSQL connection string
# DATABASE_URL="postgresql://postgres:password@localhost:5432/woof"
nano packages/database/.env

# Generate Prisma client
pnpm --filter @woof/database db:generate

# Run migrations
pnpm --filter @woof/database db:migrate

# Seed demo data (3 users, 3 pets, activities, posts)
pnpm --filter @woof/database db:seed
```

### 4. Start Development
```bash
# Start all services
pnpm dev
```

---

## 🎯 What You Get

### Monorepo Structure ✅
```
woof/
├── packages/
│   ├── ui/         → Galaxy-dark brand system
│   ├── database/   → Prisma + PostgreSQL + pgvector
│   └── config/     → Shared TypeScript/ESLint configs
├── apps/           → (Coming: web, api, mobile)
└── infra/          → Docker + CI/CD
```

### Brand System ✅
- **Colors**: #0B1C3D (primary), #6BA8FF (accent)
- **Fonts**: Space Grotesk (headings), Inter (body)
- **Logo**: Asymmetric neuron SVG
- **Theme**: Glassmorphism + dark mode

### Database Schema ✅
- 15 models (User, Pet, Activity, Post, etc.)
- pgvector support for ML compatibility
- Complete seed data with demos
- Prisma Studio available

---

## 📦 Package Commands

### Database
```bash
pnpm --filter @woof/database db:studio    # Open Prisma Studio
pnpm --filter @woof/database db:seed      # Re-seed data
pnpm --filter @woof/database db:reset     # Reset database
```

### Development
```bash
pnpm dev          # Start all apps
pnpm build        # Build all packages
pnpm lint         # Lint everything
pnpm clean        # Clean builds
```

---

## 🎨 Using the Brand System

```typescript
// In any package
import { colors, fonts, theme } from '@woof/ui/theme';

// Use colors
const primaryColor = colors.primary.DEFAULT;  // #0B1C3D

// Use fonts
const heading = fonts.heading;  // Space Grotesk

// Use glassmorphism
const glass = theme.glassmorphism.background;  // rgba(14, 18, 32, 0.7)
```

---

## 🗄️ Database Access

```typescript
// In any package
import { prisma } from '@woof/database';

// Query users
const users = await prisma.user.findMany();

// Create pet
const pet = await prisma.pet.create({
  data: {
    name: 'Buddy',
    species: 'DOG',
    breed: 'Golden Retriever',
    ownerId: user.id,
  },
});
```

---

## 🐳 Docker (Optional)

```bash
# Start PostgreSQL + Redis + n8n
docker compose -f infra/docker-compose.yml up -d

# Stop services
docker compose -f infra/docker-compose.yml down
```

---

## ✅ Verify Setup

### 1. Check Prisma Client
```bash
pnpm --filter @woof/database db:studio
# Should open Prisma Studio at http://localhost:5555
```

### 2. Check Demo Data
You should see:
- 3 Users (petlover_nyc, dogmom_sf, adventures_with_max)
- 3 Pets (Buddy, Luna, Max)
- Sample activities, posts, likes, comments

### 3. Check Brand Assets
```bash
ls packages/ui/src/theme/
# Should show: colors.ts, typography.ts, spacing.ts, logo.svg, index.ts
```

---

## 🔧 Troubleshooting

### pnpm not found
```bash
npm install -g pnpm
```

### Database connection error
1. Make sure PostgreSQL is running
2. Check `packages/database/.env` has correct URL
3. Test connection: `psql $DATABASE_URL`

### Prisma client not generated
```bash
pnpm --filter @woof/database db:generate
```

### Port already in use
```bash
# Kill process on port
lsof -ti:5555 | xargs kill -9
```

---

## 📚 Next Steps

1. **Read**: [MIGRATION_PLAN.md](./MIGRATION_PLAN.md)
2. **Check**: [PROGRESS.md](./PROGRESS.md)
3. **Build**: Start with NestJS API (apps/api)

---

## 🎯 Current Status

✅ Monorepo foundation
✅ Brand system (galaxy-dark)
✅ Database schema (15 models)
✅ Shared configs
⏳ NestJS API (next)
⏳ Next.js migration
⏳ Expo mobile app

**We're ~30% through Phase 1-3!**

---

Need help? Check [README_NEW.md](./README_NEW.md) for full documentation.

🐾 **Happy coding!**
