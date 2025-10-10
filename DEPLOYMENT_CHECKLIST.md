# 🚀 Deployment Checklist - SF Beta Launch

**Target Date**: [Set Date]
**Status**: Pre-Launch Preparation

---

## Pre-Deployment

### Code & Repository

- [ ] All PRs merged to `main` branch
- [ ] Version bumped to `1.0.0-beta`
- [ ] CHANGELOG.md updated with all features
- [ ] No critical bugs in issue tracker
- [ ] Code reviewed and approved
- [ ] All tests passing (unit, integration, e2e)

### Database

- [ ] Production database created (Supabase/Railway/custom)
- [ ] Migrations tested on staging
- [ ] Migrations applied to production
- [ ] Seed data script prepared (SF-specific)
- [ ] Backup strategy configured (automated daily)
- [ ] Connection pooling configured (PgBouncer)

### Environment Variables

#### Backend (`apps/api/.env`)
- [ ] `DATABASE_URL` - Production PostgreSQL URL
- [ ] `JWT_SECRET` - Strong random secret (32+ chars)
- [ ] `JWT_REFRESH_SECRET` - Different from JWT_SECRET
- [ ] `REDIS_URL` - Production Redis instance
- [ ] `VAPID_PUBLIC_KEY` - Web Push public key
- [ ] `VAPID_PRIVATE_KEY` - Web Push private key
- [ ] `N8N_WEBHOOK_SECRET` - Random secret for n8n auth
- [ ] `CORS_ORIGIN` - Production frontend URL
- [ ] `NODE_ENV=production`
- [ ] `PORT=4000`

#### Frontend (`apps/web/.env.local`)
- [ ] `NEXT_PUBLIC_API_URL` - Production API URL
- [ ] `NEXT_PUBLIC_VAPID_PUBLIC_KEY` - Web Push public key
- [ ] `SENTRY_DSN` - Error tracking (optional)

### Infrastructure

- [ ] Docker images built and tagged
- [ ] PostgreSQL 15+ with pgvector extension
- [ ] Redis 7+ instance running
- [ ] n8n container deployed and accessible
- [ ] SSL certificates installed (Let's Encrypt)
- [ ] CDN configured (Cloudflare/Vercel Edge)
- [ ] Load balancer configured (if needed)

---

## Deployment Steps

### 1. Deploy Database

```bash
# Apply migrations to production
cd apps/api
DATABASE_URL="postgresql://..." pnpm prisma migrate deploy

# Seed SF data
DATABASE_URL="postgresql://..." pnpm prisma db seed
```

### 2. Deploy Backend (API)

**Option A: Docker**
```bash
# Build image
docker build -t woof-api:1.0.0-beta -f apps/api/Dockerfile .

# Run container
docker run -d \
  --name woof-api \
  -p 4000:4000 \
  --env-file apps/api/.env.production \
  woof-api:1.0.0-beta

# Check health
curl http://localhost:4000/api/v1/health
```

**Option B: Railway/Render**
```bash
# Push to main branch (auto-deploys)
git push origin main

# Or use CLI
railway up
```

### 3. Deploy Frontend (Web)

**Option A: Vercel (Recommended)**
```bash
# Install Vercel CLI
pnpm add -g vercel

# Deploy
cd apps/web
vercel --prod

# Expected output: https://woof-sf-beta.vercel.app
```

**Option B: Docker**
```bash
# Build Next.js app
cd apps/web
pnpm build

# Build Docker image
docker build -t woof-web:1.0.0-beta -f Dockerfile .

# Run container
docker run -d \
  --name woof-web \
  -p 3000:3000 \
  --env-file .env.production \
  woof-web:1.0.0-beta
```

### 4. Deploy n8n Workflows

```bash
# Start n8n container
docker-compose up -d n8n

# Access n8n UI
open http://your-domain.com:5678

# Import workflows from n8n-workflows/*.json
# 1. Service Booking Follow-up
# 2. Meetup Feedback Reminder
# 3. Event Reminder
# 4. Fitness Goal Achievement

# Activate all workflows
# Configure credentials (Woof API auth)
```

### 5. Configure DNS

- [ ] Point `api.woof.app` to backend server IP
- [ ] Point `app.woof.app` to frontend (Vercel/Cloudflare)
- [ ] Point `n8n.woof.app` to n8n instance (optional)
- [ ] Verify SSL certificates auto-provisioned
- [ ] Test HTTPS on all subdomains

---

## Post-Deployment Verification

### API Health Checks

```bash
# Health endpoint
curl https://api.woof.app/api/v1/health
# Expected: { "status": "ok", "timestamp": "..." }

# Auth endpoint
curl https://api.woof.app/api/v1/auth/me \
  -H "Authorization: Bearer [TEST_JWT]"
# Expected: User object or 401

# Analytics
curl https://api.woof.app/api/v1/analytics/north-star?timeframe=7d
# Expected: North star metrics
```

### Frontend Checks

- [ ] Visit https://app.woof.app
- [ ] Register new account
- [ ] Complete onboarding flow
- [ ] Browse matches
- [ ] RSVP to event
- [ ] Check console for errors (should be 0)

### Push Notifications

- [ ] Enable notifications in settings
- [ ] Verify subscription stored in database
- [ ] Trigger test notification from backend
- [ ] Receive notification on device
- [ ] Click notification → opens app

### n8n Workflows

- [ ] Service booking 24h follow-up active
- [ ] Meetup feedback reminder running every 30min
- [ ] Event reminder running every hour
- [ ] Fitness goal checker running daily at 9 PM PST
- [ ] Check execution logs (no errors)

---

## Monitoring Setup

### Error Tracking (Sentry)

```bash
# Install Sentry SDK (already in package.json)
# Add to apps/api/src/main.ts and apps/web/src/app/layout.tsx

# Test error tracking
curl -X POST https://api.woof.app/api/v1/test/error
# Expected: Error appears in Sentry dashboard
```

### Analytics Dashboard

- [ ] Access `/analytics/north-star`
- [ ] Verify metrics are populating
- [ ] Set up Grafana/Metabase (optional)
- [ ] Configure alerts for:
  - Error rate >1%
  - API latency >500ms
  - Database connections >80%

### Uptime Monitoring

- [ ] Add to UptimeRobot / Pingdom
- [ ] Monitor endpoints:
  - `https://api.woof.app/api/v1/health`
  - `https://app.woof.app`
- [ ] Alert email: ops@woof.app
- [ ] Alert Slack channel: #woof-alerts

---

## Beta Tester Setup

### Invite List (50 SF Dog Owners)

- [ ] Import contacts to mailing list
- [ ] Prepare welcome email with:
  - App URL: https://app.woof.app
  - Login instructions
  - Onboarding video (optional)
  - Feedback form link

### Email Template

```
Subject: 🐾 You're invited to Woof SF Beta!

Hi [Name],

You're one of 50 selected beta testers for Woof - San Francisco's first
dog social fitness platform!

Get Started:
1. Visit: https://app.woof.app
2. Sign up with this email
3. Add your dog's profile
4. Discover compatible pups nearby!

Features to Try:
🐕 ML-powered dog matching
📅 Join community events at Golden Gate Park, Dolores Park, and more
🏆 Earn points and badges for activity
💬 Chat with other dog owners
🔔 Get proactive meetup suggestions

We Need Your Feedback:
This is a beta, so your input is invaluable. Please report any bugs or
suggestions to: beta@woof.app

SF Beta Perks:
✨ Lifetime "Founding Member" badge
✨ Early access to new features
✨ Free entry to exclusive SF dog events

Let's make San Francisco the most dog-friendly city!

Woof Team
https://woof.app
```

### Beta Group Management

- [ ] Create private Discord/Slack channel
- [ ] Share beta tester guidelines
- [ ] Schedule kickoff Zoom call (optional)
- [ ] Set up weekly check-in cadence

---

## Rollback Plan

### Database Rollback

```bash
# Revert last migration
cd apps/api
DATABASE_URL="postgresql://..." pnpm prisma migrate resolve --rolled-back [migration-name]
```

### Application Rollback

**Vercel**:
```bash
# List deployments
vercel ls

# Rollback to previous
vercel rollback [deployment-url]
```

**Docker**:
```bash
# Stop current containers
docker stop woof-api woof-web

# Start previous version
docker run -d --name woof-api woof-api:0.9.0
docker run -d --name woof-web woof-web:0.9.0
```

---

## Success Metrics (Week 1)

- [ ] 50+ beta testers signed up
- [ ] 100+ pet profiles created
- [ ] 20+ meetups proposed
- [ ] 5+ events with RSVPs
- [ ] 0 critical bugs reported
- [ ] <1% error rate
- [ ] API uptime >99.5%
- [ ] Average page load <2s

---

## Communication Channels

### Internal

- **Slack**: #woof-launch
- **Email**: team@woof.app
- **On-call**: [Phone Number]

### External

- **Support**: beta@woof.app
- **Social**: @woofapp (Twitter/Instagram)
- **Feedback**: https://forms.gle/...

---

## Post-Launch Actions (Day 2-7)

- [ ] Daily check-in with beta testers
- [ ] Monitor error logs (Sentry)
- [ ] Review analytics dashboard
- [ ] Collect feedback via survey
- [ ] Prioritize bug fixes
- [ ] Plan feature iterations
- [ ] Prepare weekly update email

---

**Deployment Lead**: [Name]
**Signed Off By**:
- [ ] Product Manager
- [ ] Engineering Lead
- [ ] QA Lead

**Deployment Date**: _______________
**Time**: _______________ (PST)

---

Good luck with the launch! 🚀🐾
