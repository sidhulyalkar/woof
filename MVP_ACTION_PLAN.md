# 🚀 Woof MVP Action Plan

**Goal**: Deploy beta-ready MVP within 2-3 days
**Current Status**: 75% ready (blocked by 35 build errors)
**Date**: October 12, 2025

---

## 🎯 Critical Path to Deployment

### **Phase 1: Fix Build Errors** ⏱️ 2-4 hours (IMMEDIATE)

#### Task 1.1: Add Missing Models to Schema (45 min)

**Location**: [packages/database/prisma/schema.prisma](packages/database/prisma/schema.prisma)

Add these models:

```prisma
model BadgeAward {
  id        String   @id @default(uuid())
  userId    String   @map("user_id")
  badgeType String   @map("badge_type")
  awardedAt DateTime @default(now()) @map("awarded_at")

  @@unique([userId, badgeType])
  @@index([userId])
  @@index([badgeType])
  @@map("badge_awards")
}

model WeeklyStreak {
  id             String   @id @default(uuid())
  userId         String   @unique @map("user_id")
  currentWeek    Int      @default(0) @map("current_week")
  lastActivityAt DateTime @map("last_activity_at")
  createdAt      DateTime @default(now()) @map("created_at")
  updatedAt      DateTime @updatedAt @map("updated_at")

  @@index([userId])
  @@map("weekly_streaks")
}
```

Update User model:
```prisma
model User {
  // ... existing fields
  totalPoints Int @default(0) @map("total_points")

  // ... existing relations
  hostedEvents CommunityEvent[] @relation("EventHost")
}
```

Update CommunityEvent model:
```prisma
model CommunityEvent {
  // ... existing fields
  organizer User @relation("EventHost", fields: [hostUserId], references: [id])
}
```

Update EventRSVP model:
```prisma
model EventRSVP {
  // ... existing fields
  user User @relation("EventRSVPs", fields: [userId], references: [id])

  @@index([userId])
}
```

Add to User model:
```prisma
model User {
  // ... existing relations
  eventRsvps EventRSVP[] @relation("EventRSVPs")
}
```

Update EventFeedback model:
```prisma
model EventFeedback {
  // ... existing fields
  user User @relation("EventFeedbacks", fields: [userId], references: [id])
}
```

Add to User model:
```prisma
model User {
  // ... existing relations
  eventFeedbacks EventFeedback[] @relation("EventFeedbacks")
}
```

**Then run:**
```bash
cd packages/database
pnpm prisma migrate dev --name add_missing_gamification_and_event_models
pnpm prisma generate
```

#### Task 1.2: Fix Event DTOs (15 min)

**Location**: [apps/api/src/events/dto/create-event.dto.ts](apps/api/src/events/dto/create-event.dto.ts)

```typescript
export class CreateEventDto {
  // ... existing fields

  @IsOptional()
  @IsDateString()
  endTime?: string;

  @IsOptional()
  @IsNumber()
  capacity?: number;
}
```

**Location**: [apps/api/src/events/events.service.ts](apps/api/src/events/events.service.ts:26)

```typescript
// Line 26 - handle optional endTime
endTime: dto.endTime ? new Date(dto.endTime) : new Date(new Date(dto.startTime).getTime() + 2 * 60 * 60 * 1000), // Default: 2 hours
```

#### Task 1.3: Fix Type Annotations (30 min)

**Install missing types:**
```bash
cd apps/api
pnpm add -D @types/web-push
```

**Fix controllers:**

[apps/api/src/notifications/notifications.controller.ts](apps/api/src/notifications/notifications.controller.ts:33):
```typescript
import { Request as ExpressRequest } from 'express';

@Delete('unsubscribe/:endpoint')
async unsubscribe(@Param('endpoint') endpoint: string, @Request() req: ExpressRequest) {
  return this.notificationsService.unsubscribePushNotification(
    (req as any).user.id,
    endpoint,
  );
}
```

[apps/api/src/nudges/nudges.controller.ts](apps/api/src/nudges/nudges.controller.ts) - same pattern for all methods.

**Fix exception filter:**

[apps/api/src/common/filters/all-exceptions.filter.ts](apps/api/src/common/filters/all-exceptions.filter.ts:49):
```typescript
user: request['user']
  ? {
      id: (request['user'] as any).id,
      email: (request['user'] as any).email,
    }
  : undefined,
```

#### Task 1.4: Fix Auth Test (5 min)

[apps/api/src/auth/auth.service.spec.ts](apps/api/src/auth/auth.service.spec.ts:103):
```typescript
const mockLoginDto = {
  email: mockUser.email,
  password: 'password123' // Add this
};

const result = await service.login(mockLoginDto);
```

#### Task 1.5: Rebuild & Verify (15 min)

```bash
cd /Users/sidhulyalkar/Documents/App_Dev/woof
pnpm build
```

**Expected**: 0 errors ✅

---

### **Phase 2: Testing** ⏱️ 4 hours (SAME DAY)

#### Task 2.1: Start Services (5 min)

```bash
# Terminal 1: API
pnpm --filter @woof/api dev

# Terminal 2: Web
pnpm --filter @woof/web dev

# Verify:
# API: http://localhost:4000/docs
# Web: http://localhost:3000
```

#### Task 2.2: Smoke Test Critical Flows (2 hours)

**Auth Flow** (30 min):
1. Register new user → Success
2. Login → Receive JWT
3. Refresh token → New JWT
4. Access protected endpoint → Success

**Nudge Flow** (45 min):
1. Create 2 users with pets
2. Set compatible profiles
3. Simulate proximity (update location pings)
4. Verify nudge created
5. Test push notification
6. Accept nudge → Navigate to chat/meetup
7. Dismiss nudge → Hide from list

**Events Flow** (45 min):
1. Create community event
2. RSVP as different user
3. Check-in at event
4. Submit feedback
5. View event with attendees

#### Task 2.3: Database Verification (30 min)

```bash
# Check all tables created
pnpm --filter @woof/database db:studio

# Verify:
- badge_awards table exists
- weekly_streaks table exists
- users.total_points column exists
- All migrations applied
```

#### Task 2.4: Browser Testing (1 hour)

Test in Chrome, Safari, Firefox:
- [ ] Registration flow
- [ ] Pet creation
- [ ] Discover matches
- [ ] Send message
- [ ] Create event
- [ ] Push notification permission
- [ ] Receive test notification

---

### **Phase 3: Deployment Prep** ⏱️ 3 hours (DAY 2)

#### Task 3.1: Environment Configuration (1 hour)

**Create production `.env` files:**

`apps/api/.env.production`:
```bash
DATABASE_URL="postgresql://user:pass@host:5432/woof_prod"
JWT_SECRET="<generate-256-bit-secret>"
JWT_EXPIRES_IN="15m"
REFRESH_SECRET="<generate-256-bit-secret>"
REFRESH_EXPIRES_IN="7d"

SENTRY_DSN="https://your-sentry-dsn"
NODE_ENV="production"
PORT=4000

VAPID_PUBLIC_KEY="<production-vapid-public>"
VAPID_PRIVATE_KEY="<production-vapid-private>"

AWS_ACCESS_KEY_ID="<your-key>"
AWS_SECRET_ACCESS_KEY="<your-secret>"
AWS_REGION="us-west-1"
AWS_BUCKET_NAME="woof-prod-uploads"

CORS_ORIGIN="https://woof.app"
```

`apps/web/.env.production`:
```bash
NEXT_PUBLIC_API_URL="https://api.woof.app"
NEXT_PUBLIC_VAPID_PUBLIC_KEY="<production-vapid-public>"
NEXT_PUBLIC_SENTRY_DSN="https://your-sentry-dsn"
```

#### Task 3.2: Database Setup (30 min)

**Provision production database** (Neon, Supabase, or Fly Postgres):

```bash
# Run migrations
DATABASE_URL="<production-url>" pnpm --filter @woof/database prisma migrate deploy

# Generate client
pnpm --filter @woof/database prisma generate

# Seed initial data
DATABASE_URL="<production-url>" pnpm --filter @woof/api db:seed
```

#### Task 3.3: Deploy Services (1.5 hours)

**Deploy API to Fly.io:**
```bash
# Install flyctl
brew install flyctl

# Login
fly auth login

# Launch app
cd apps/api
fly launch --name woof-api-prod

# Set secrets
fly secrets set DATABASE_URL="<url>" JWT_SECRET="<secret>" --app woof-api-prod

# Deploy
fly deploy
```

**Deploy Web to Vercel:**
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
cd apps/web
vercel --prod
```

---

### **Phase 4: Beta Launch** ⏱️ 1 day (DAY 3)

#### Task 4.1: Staging Verification (2 hours)

Test all critical flows on production URLs:
- [ ] User registration
- [ ] Pet creation
- [ ] Discover & matching
- [ ] Create/join events
- [ ] Send messages
- [ ] Push notifications
- [ ] Service discovery

#### Task 4.2: Monitoring Setup (1 hour)

**Sentry Verification:**
- [ ] API errors reporting
- [ ] Web errors reporting
- [ ] Source maps uploaded
- [ ] Alerts configured

**Database Monitoring:**
- [ ] Connection pooling configured
- [ ] Slow query logging enabled
- [ ] Backup schedule confirmed

**Custom Metrics:**
```typescript
// Add to analytics service
trackEvent('APP_OPEN', { userId, timestamp });
trackEvent('NUDGE_ACCEPTED', { nudgeId, userId });
trackEvent('MEETUP_OCCURRED', { proposalId, rating });
```

#### Task 4.3: Invite Beta Testers (3 hours)

**Closed Beta Cohort** (10-20 SF users):
1. Create invitation system
2. Send personalized invites
3. Provide onboarding guide
4. Set up feedback channel (Discord/Slack)
5. Schedule check-in calls

**Day 1-7 Monitoring:**
- Daily active users
- Nudge acceptance rate
- Meetup proposals created
- Events created/attended
- Error rates
- Support requests

---

## 📋 Quick Reference Checklist

### Pre-Deployment
- [ ] All build errors fixed (35 → 0)
- [ ] Database schema updated
- [ ] Migrations applied locally
- [ ] Smoke tests passing
- [ ] Environment variables configured
- [ ] Service worker deployed

### Deployment
- [ ] Production database provisioned
- [ ] Migrations applied to prod
- [ ] API deployed to Fly.io
- [ ] Web deployed to Vercel
- [ ] DNS configured
- [ ] SSL certificates verified

### Post-Deployment
- [ ] Staging tests passing
- [ ] Sentry error tracking live
- [ ] Analytics tracking active
- [ ] Beta invites sent
- [ ] Monitoring dashboard set up
- [ ] Feedback channel created

---

## 🚨 Emergency Contacts & Resources

### If Build Fails:
1. Check Prisma client regeneration: `pnpm --filter @woof/database prisma generate`
2. Clear build cache: `pnpm clean && pnpm install`
3. Review build logs: `pnpm build 2>&1 | tee build.log`

### If Database Fails:
1. Check connection: `docker ps` (ensure woof-postgres running)
2. Reset database: `docker-compose down -v && docker-compose up -d`
3. Reapply migrations: `pnpm --filter @woof/database prisma migrate dev`

### If Deployment Fails:
- **Fly.io**: `fly logs --app woof-api-prod`
- **Vercel**: Check deployment logs in dashboard
- **Database**: Verify connection string, check firewall rules

---

## 📊 Success Criteria

### Phase 1 Complete When:
✅ `pnpm build` shows 0 errors
✅ All tests pass
✅ API starts without errors
✅ Web builds successfully

### Phase 2 Complete When:
✅ Auth flow works end-to-end
✅ Nudge engine triggers correctly
✅ Push notifications delivered
✅ Events flow functional

### Phase 3 Complete When:
✅ API accessible at production URL
✅ Web accessible at production URL
✅ Database responding to queries
✅ Sentry capturing errors

### Phase 4 Complete When:
✅ 10+ beta users invited
✅ First real-world meetup occurs
✅ Feedback channel active
✅ Metrics dashboard populated

---

## 🎯 Timeline Summary

| Phase | Duration | Blocking | Start |
|-------|----------|----------|-------|
| **Fix Build Errors** | 2-4 hours | YES | Immediate |
| **Testing** | 4 hours | No | Same day |
| **Deployment Prep** | 3 hours | No | Day 2 |
| **Beta Launch** | 1 day | No | Day 3 |

**Total Time to Beta**: 2-3 days

---

## 💡 Pro Tips

1. **Tackle build errors methodically**: Start with schema changes, then services, then controllers
2. **Test after each fix**: Don't wait until all fixes are done
3. **Use Prisma Studio**: Visual database tool is invaluable for debugging
4. **Monitor Sentry closely**: First 48 hours will reveal production issues
5. **Engage beta users early**: Their feedback is gold

---

## 📚 Additional Resources

- [MVP_READINESS_REPORT.md](MVP_READINESS_REPORT.md) - Comprehensive technical evaluation
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [NUDGE_ENGINE_SETUP.md](NUDGE_ENGINE_SETUP.md) - Nudge system documentation
- [SESSION_SUMMARY.md](SESSION_SUMMARY.md) - Recent implementation history

---

**Ready to deploy?** Start with Phase 1, Task 1.1! 🚀

*Action Plan Created: October 12, 2025*
*Estimated Completion: October 14-15, 2025*
