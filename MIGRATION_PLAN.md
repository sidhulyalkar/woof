# Woof Migration Plan - Hybrid Approach

## Overview
Migrating from flat repo structure (FastAPI + Next.js) to production-grade monorepo (NestJS + Next.js + Expo) following woofAgentSpec.

## Architecture Changes

### Before (Current)
```
woof/
├── frontend/          # Next.js 15 (npm)
├── backend/           # FastAPI (Python)
├── ml/                # PyTorch models
├── infra/             # Docker configs
├── n8n/               # Workflows
└── docs/
```

### After (Target)
```
woof/
├── apps/
│   ├── web/           # Next.js 15 (migrated from frontend/)
│   ├── mobile/        # Expo React Native (NEW)
│   ├── api/           # NestJS (replaces backend/)
│   └── automations/   # n8n (migrated)
├── packages/
│   ├── ui/            # Shared components + brand system
│   ├── database/      # Prisma schema + client
│   ├── config/        # Shared configs (TS, ESLint, etc)
│   └── sdk/           # Generated OpenAPI SDK
├── infra/
│   ├── docker/        # Docker configs
│   ├── db/            # Database migrations & seeds
│   └── ci/            # GitHub Actions
├── docs/
└── pnpm-workspace.yaml
```

## Migration Strategy - 6 Phases

### Phase 1: Monorepo Foundation (Day 1-2)
**Goal**: Set up monorepo structure without breaking existing code

- [ ] Install pnpm globally
- [ ] Create pnpm-workspace.yaml
- [ ] Set up Turborepo (turbo.json)
- [ ] Create packages/config (tsconfig, eslint)
- [ ] Move frontend → apps/web
- [ ] Update all imports and paths
- [ ] Verify apps/web runs correctly

**Validation**: `pnpm dev` runs Next.js successfully

---

### Phase 2: Brand System & Shared UI (Day 2-3)
**Goal**: Create galaxy-dark design system

- [ ] Create packages/ui with Tailwind config
- [ ] Define brand tokens (colors, typography, spacing)
- [ ] Generate asymmetric neuron logo (SVG exports)
- [ ] Build core UI components (Button, Card, Input)
- [ ] Apply galaxy-dark theme to apps/web
- [ ] Implement glassmorphism utilities

**Validation**: apps/web displays new brand

---

### Phase 3: Database Layer (Day 3-4)
**Goal**: Replace Supabase with Prisma + PostgreSQL + pgvector

- [ ] Create packages/database
- [ ] Define Prisma schema (User, Pet, Activity, etc.)
- [ ] Set up PostgreSQL with pgvector extension
- [ ] Create seed data script
- [ ] Generate Prisma Client
- [ ] Write database utilities (connection, helpers)

**Validation**: `pnpm db:migrate` and `pnpm db:seed` work

---

### Phase 4: NestJS API (Day 4-7)
**Goal**: Replace FastAPI with production-grade NestJS backend

- [ ] Initialize NestJS in apps/api
- [ ] Install dependencies (Prisma, Passport, BullMQ, Socket.io)
- [ ] Create module structure (auth, users, pets, activities, social)
- [ ] Implement JWT authentication + refresh tokens
- [ ] Build core endpoints (matching api-spec.md)
- [ ] Add Swagger/OpenAPI documentation
- [ ] Integrate packages/database
- [ ] Set up BullMQ for background jobs
- [ ] Add Socket.io for real-time features

**Validation**: Swagger docs at localhost:4000/api

---

### Phase 5: ML & Advanced Features (Day 7-9)
**Goal**: Port ML models and add pgvector compatibility

- [ ] Create ML endpoints in apps/api
- [ ] Port compatibility scoring to pgvector
- [ ] Implement energy prediction model
- [ ] Create background workers for model inference
- [ ] Add caching layer (Redis)
- [ ] Implement meetup suggestion algorithm

**Validation**: `/api/compatibility/predict` returns scores

---

### Phase 6: Mobile App & Infrastructure (Day 9-12)
**Goal**: Add Expo mobile app and finalize infra

- [ ] Initialize Expo in apps/mobile
- [ ] Set up Expo Router + navigation
- [ ] Integrate HealthKit/Google Fit
- [ ] Build core mobile screens
- [ ] Generate OpenAPI SDK in packages/sdk
- [ ] Update Docker Compose for all services
- [ ] Configure GitHub Actions CI/CD
- [ ] Create build artifact script
- [ ] Write deployment docs

**Validation**: Mobile app connects to API, CI passes

---

## Migration Checklist

### Immediate (Week 1)
- [x] Document migration plan
- [ ] Backup current codebase
- [ ] Set up monorepo structure
- [ ] Create shared packages
- [ ] Migrate frontend to apps/web
- [ ] Build brand system
- [ ] Set up Prisma database

### Short-term (Week 2)
- [ ] Complete NestJS API
- [ ] Implement authentication
- [ ] Port core endpoints
- [ ] Add Socket.io real-time features
- [ ] Integrate ML models

### Medium-term (Week 3)
- [ ] Build Expo mobile app
- [ ] Set up CI/CD
- [ ] Docker infrastructure
- [ ] Testing suite
- [ ] Deployment automation

## Rollback Plan
If migration fails at any phase:
1. Current code preserved in `/backup` branch
2. Each phase is a separate git branch
3. Can revert to previous working state
4. Frontend always remains functional

## Success Metrics
- ✅ One-command dev setup: `pnpm dev`
- ✅ All services running in Docker
- ✅ Swagger docs auto-generated
- ✅ Mobile app builds successfully
- ✅ CI/CD passes all checks
- ✅ Build artifact created: `Woof_v1.0.0.zip`

## Tech Stack Changes

| Component | Before | After | Reason |
|-----------|--------|-------|--------|
| Backend | FastAPI | NestJS | TypeScript, better DX, scaling |
| Database | Supabase | Prisma + PostgreSQL | Full control, pgvector |
| Package Manager | npm | pnpm | Speed, disk efficiency |
| Monorepo | None | Turborepo | Caching, parallel builds |
| Mobile | None | Expo | Cross-platform, HealthKit |
| Auth | Supabase Auth | Passport.js | Custom providers |

## Timeline
- **Week 1**: Foundation + Database + Brand
- **Week 2**: API + Auth + Core Features
- **Week 3**: Mobile + Infra + CI/CD + Polish

**Total**: ~15-20 days to production-ready MVP

---

Generated: 2025-10-01
Version: 1.0.0
Status: IN PROGRESS
