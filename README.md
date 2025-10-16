# 🐾 Woof - Pet Social Fitness Platform

> **Production-ready social network** for dog owners to discover compatible playmates, coordinate meetups, and track activities together.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
[![NestJS](https://img.shields.io/badge/NestJS-10-red)](https://nestjs.com/)
[![pnpm](https://img.shields.io/badge/pnpm-8.15-orange)](https://pnpm.io/)

**Status**: ✅ Production Ready | **Target**: San Francisco Beta Launch

---

## 🎯 What is Woof?

Woof brings dog owners together through **IRL meetups** backed by smart compatibility matching and activity tracking.

### Core Features

- **🐕 Pet Profiles** - Detailed profiles with temperament, energy level, play style
- **🤝 Compatibility Matching** - ML-powered recommendations based on pet personalities
- **📅 Event Coordination** - Create and join dog meetups at local parks
- **📊 Activity Tracking** - Log walks, runs, playtime with photos and metrics
- **💬 Real-time Chat** - Coordinate meetups with instant messaging
- **🏆 Gamification** - Points, badges, and leaderboards for engagement
- **🏪 Service Discovery** - Find trainers, groomers, vets, and daycares
- **✅ Verification** - Trust badges for verified profiles and businesses

### What Makes Woof Different

Unlike Rover (services) or BarkHappy (basic social), Woof focuses on:
- **IRL Meetup Outcomes** - Track if dogs actually got along
- **Quality Metrics** - "Great match!", "Needs slow intro" labels
- **Activity Data** - Co-activity patterns and repeat meetup rates
- **Service Conversions** - Meetups → trainer bookings pipeline
- **ML-Ready Architecture** - Systematic data collection for better matching

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 20+
- **pnpm** 8+
- **Docker** (for local PostgreSQL)

### Local Development

```bash
# 1. Install dependencies
pnpm install

# 2. Start PostgreSQL
docker compose up -d

# 3. Set up database
cp apps/api/.env.example apps/api/.env
pnpm --filter @woof/api prisma generate
pnpm --filter @woof/api prisma migrate dev
pnpm --filter @woof/api db:seed

# 4. Start development servers
cp apps/web/.env.local.example apps/web/.env.local

# Terminal 1: API
pnpm --filter @woof/api dev

# Terminal 2: Web
pnpm --filter @woof/web dev

# Terminal 3: Mobile (optional)
pnpm --filter @woof/mobile start
```

**Endpoints**:
- Frontend: http://localhost:3000
- API: http://localhost:4000
- API Docs: http://localhost:4000/docs
- Mobile: Expo DevTools at http://localhost:8081

### Test Credentials

All seed users have password: `password123`

Example logins:
- `sarah@example.com` - Software engineer with Golden Retriever
- `mike@example.com` - Finance professional with Rescue Mix
- `jen@example.com` - Graphic designer with Corgi

---

## 🏗️ Architecture

### Monorepo Structure

```
woof/
├── apps/
│   ├── api/              # NestJS backend (18 modules)
│   ├── web/              # Next.js 15 frontend (158+ components)
│   └── mobile/           # React Native mobile app (iOS & Android)
├── packages/
│   ├── database/         # Prisma schema (PostgreSQL + pgvector)
│   ├── ui/               # Shared UI components
│   └── config/           # TypeScript/ESLint configs
└── .github/workflows/    # CI/CD (test, build, deploy)
```

### Tech Stack

**Backend**:
- NestJS 10 with TypeScript
- PostgreSQL with Prisma ORM
- pgvector for ML embeddings
- Socket.io for real-time chat
- AWS S3/Cloudflare R2 for file storage
- JWT authentication with refresh tokens

**Web Frontend**:
- Next.js 15 with App Router
- React 19 with TypeScript
- Tailwind CSS v4
- Zustand for state management
- React Query for server state
- PWA with offline support

**Mobile App**:
- React Native with Expo SDK 54
- React Navigation 7
- TypeScript 5.9
- Axios with JWT auto-refresh
- Native features: Camera, Maps, Location, Notifications

**Infrastructure**:
- GitHub Actions CI/CD
- Sentry error tracking
- Vercel deployment (web)
- Fly.io deployment (API)
- EAS Build (mobile)
- Neon/Supabase (PostgreSQL)

---

## 📦 Key Modules

### Backend (18 Modules)

1. **Auth** - JWT with refresh tokens
2. **Users** - Profile management
3. **Pets** - Pet profiles with ML embeddings
4. **Activities** - Walk/run/play tracking
5. **Social** - Posts, likes, comments
6. **Meetups** - Event coordination
7. **Compatibility** - ML-powered matching
8. **Events** - Community events with check-ins
9. **Gamification** - Points, badges, leaderboards
10. **Services** - Business directory
11. **Verification** - Profile verification
12. **Analytics** - North star metrics
13. **Co-Activity** - Shared activity tracking
14. **Meetup Proposals** - Direct meetup invites
15. **Storage** - S3/R2 file uploads
16. **Chat** - Real-time messaging
17. **Manual Activities** - Activity logging with photos
18. **... and more**

### Frontend (158+ Components)

- **Auth**: Login, register, onboarding wizard
- **Activity**: Manual logging, activity feed
- **Discover**: Match discovery, compatibility cards
- **Events**: Event creation, check-ins, attendance
- **Feed**: Social posts, likes, comments
- **Gamification**: Badges, leaderboards, achievements
- **Inbox**: Real-time messaging
- **Profile**: User/pet profiles, edit forms
- **Services**: Service discovery, filters, bookings
- **UI Library**: 50+ Radix UI components with custom styling

---

## 🧪 Testing

### Test Coverage

- **Backend**: 80%+ target (Jest + Supertest)
- **Frontend**: 70%+ target (Vitest + Playwright)
- **E2E**: Critical user flows (auth, events, messaging)

### Run Tests

```bash
# Backend tests
pnpm --filter @woof/api test
pnpm --filter @woof/api test:e2e

# Frontend tests
pnpm --filter @woof/web test
pnpm --filter @woof/web test:e2e
pnpm --filter @woof/web test:e2e:ui

# All tests
pnpm test
```

### CI/CD

Three automated workflows:
1. **CI**: Lint → Test → Build (on every push)
2. **Deploy Staging**: Auto-deploy to staging (on develop branch)
3. **Deploy Production**: Deploy to production (on main branch)

---

## 🌉 San Francisco Beta Data

The seed script generates realistic SF-focused test data:

### Included in Seed Data

- **20 Users** across 18 SF neighborhoods
- **20 Pets** with popular SF breeds (Goldens, Frenchies, Corgis, Aussies)
- **12 Dog Parks** with GPS coordinates (Fort Funston, Crissy Field, Dolores, etc.)
- **5 Upcoming Events** at iconic SF locations
- **5 Pet Services** (training, grooming, daycare)
- **50 Activity Logs** from past 30 days
- **30 Social Posts** tagged at real parks

### Run Seed Script

```bash
pnpm --filter @woof/api db:seed
```

See [apps/api/prisma/SEED_DATA_README.md](apps/api/prisma/SEED_DATA_README.md) for details.

---

## 🔒 Security

Production-grade security measures:

- **Rate Limiting**: 3 tiers (3/sec, 20/10s, 100/min)
- **Helmet**: CSP, XSS, clickjacking protection
- **CORS**: Strict origin validation
- **JWT**: Secure authentication with refresh tokens
- **File Upload**: Size limits, type validation
- **Content Security Policy**: Nonce-based scripts
- **HTTPS**: Forced redirect in production
- **Error Tracking**: Sentry with session replay

---

## 📊 Monitoring & Analytics

### Error Tracking

- **Sentry**: Backend + frontend integration
- **Session Replay**: Debug user issues
- **Performance Monitoring**: Track API response times
- **Error Boundaries**: User-friendly fallbacks

### Analytics

- **Vercel Analytics**: Web vitals
- **Custom Events**: North star metrics
  - Successful IRL meetups
  - Repeat meetup rate
  - Service conversions
  - Event attendance
  - Co-activity frequency

---

## 🚢 Deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete instructions.

### Quick Deploy

```bash
# 1. Set environment variables
# See .env.example files

# 2. Deploy API
fly deploy

# 3. Deploy Web
vercel deploy --prod

# 4. Run migrations
pnpm --filter @woof/api prisma migrate deploy

# 5. Seed data
pnpm --filter @woof/api db:seed
```

### Environment Variables

**Backend** (apps/api/.env):
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET` - JWT signing key
- `SENTRY_DSN` - Error tracking
- `S3_*` - File storage credentials
- `CORS_ORIGIN` - Allowed origins

**Frontend** (apps/web/.env.local):
- `NEXT_PUBLIC_API_URL` - API endpoint
- `NEXT_PUBLIC_SENTRY_DSN` - Error tracking
- `SENTRY_AUTH_TOKEN` - Source map upload

---

## 📚 Documentation

- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Production deployment
- **[ML_SYSTEM_README.md](ML_SYSTEM_README.md)** - ML compatibility system
- **[apps/api/prisma/SEED_DATA_README.md](apps/api/prisma/SEED_DATA_README.md)** - Seed data guide
- **[/docs](http://localhost:4000/docs)** - API Swagger documentation (when running)

---

## 🛠️ Development Scripts

```bash
# Development
pnpm dev                    # Start all apps
pnpm --filter @woof/api dev # Start API only
pnpm --filter @woof/web dev # Start web only

# Database
pnpm db:migrate            # Run migrations
pnpm db:seed               # Seed data
pnpm db:studio             # Open Prisma Studio

# Testing
pnpm test                  # Run all tests
pnpm test:coverage         # Test with coverage
pnpm test:e2e              # E2E tests

# Building
pnpm build                 # Build all packages
pnpm lint                  # Lint all code
pnpm type-check            # TypeScript checks
```

---

## 🗺️ Roadmap

### ✅ Completed (Beta Ready)

- Core authentication & authorization
- Pet profiles with compatibility matching
- Event creation & management
- Real-time messaging
- Activity tracking with photos
- File upload system
- Service discovery
- Gamification system
- Comprehensive testing
- CI/CD automation
- Security hardening
- Error tracking
- San Francisco seed data

### 🚀 Post-Beta (Phase 3)

- Push notifications
- n8n automation workflows
- Advanced communities/groups
- Calendar integration
- Enhanced ML algorithm
- Referral system
- Premium features
- Mobile apps (React Native)
- Video calls
- Stories/highlights

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Standards**:
- TypeScript strict mode
- ESLint with 0 errors
- Prettier formatting
- Conventional commits
- 70%+ test coverage

---

## 📄 License

MIT © [Sidharth Hulyalkar](https://github.com/sidhulyalkar)

---

## 🙏 Acknowledgments

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [NestJS](https://nestjs.com/) - Node.js framework
- [Prisma](https://www.prisma.io/) - Database ORM
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Radix UI](https://www.radix-ui.com/) - Accessible components
- [Sentry](https://sentry.io/) - Error tracking
- And many more amazing open source projects

---

**Woof** - Bringing dog owners together, one park at a time 🐾🌉
