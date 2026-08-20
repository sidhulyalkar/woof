# 🧪 Beta Testing Guide - San Francisco Launch

**Version**: 1.0.0
**Status**: Ready for SF Beta
**Last Updated**: 2025-10-10

## 📋 Table of Contents

- [Overview](#overview)
- [Test Environment Setup](#test-environment-setup)
- [Functional Testing Checklist](#functional-testing-checklist)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [Analytics Validation](#analytics-validation)
- [Known Issues](#known-issues)
- [Reporting Bugs](#reporting-bugs)

---

## Overview

This guide outlines the comprehensive testing plan for the Woof SF Beta launch. All features listed below must be tested and verified before launch.

### Testing Scope

- ✅ **User Registration & Onboarding**
- ✅ **Pet Profile & Quiz System**
- ✅ **Match Discovery & Compatibility**
- ✅ **Meetup Proposals & Coordination**
- ✅ **Event System (RSVP, Check-in, Feedback)**
- ✅ **Service Discovery & Intent Tracking**
- ✅ **Gamification (Points & Badges)**
- ✅ **Push Notifications**
- ✅ **Proactive Nudges**
- ✅ **Real-time Chat**

---

## Test Environment Setup

### Prerequisites

1. **Database**: PostgreSQL with pgvector extension
2. **Redis**: For caching and queues
3. **n8n**: Workflow automation (optional for initial testing)

### Local Setup

```bash
# 1. Start infrastructure services
docker-compose up -d postgres redis n8n

# 2. Install dependencies
pnpm install

# 3. Setup environment variables
cp apps/api/.env.example apps/api/.env
cp apps/web/.env.local.example apps/web/.env.local

# 4. Run database migrations
cd apps/api
pnpm prisma migrate deploy
pnpm prisma db seed

# 5. Start development servers
# Terminal 1 - API
cd apps/api
pnpm dev

# Terminal 2 - Web
cd apps/web
pnpm dev
```

### Test Data

Run the seed script to populate SF-specific test data:

```bash
cd apps/api
pnpm prisma db seed
```

This creates:
- 20 SF-based test users
- 30+ pets with complete profiles
- 10+ community events in SF locations
- 50+ service providers (groomers, vets, trainers, etc.)
- Sample compatibility edges and quiz responses

---

## Functional Testing Checklist

### 1. User Registration & Authentication

#### Test Case 1.1: New User Registration
- [ ] Navigate to `/login`
- [ ] Click "Sign Up"
- [ ] Fill in: Email, Password, Handle
- [ ] Submit form
- [ ] **Expected**: Redirected to onboarding, JWT token stored

#### Test Case 1.2: Login
- [ ] Navigate to `/login`
- [ ] Enter valid credentials
- [ ] Click "Log In"
- [ ] **Expected**: Redirected to home, authenticated

#### Test Case 1.3: Logout
- [ ] Click profile → "Log Out"
- [ ] **Expected**: Redirected to `/login`, session cleared

### 2. Onboarding Flow

#### Test Case 2.1: Pet Profile Creation
- [ ] After registration, start onboarding
- [ ] Click "Add Your Dog"
- [ ] Fill in: Name, Breed, Age, Gender
- [ ] Upload avatar (test with 5MB+ image)
- [ ] Submit
- [ ] **Expected**: Pet created, moved to quiz

#### Test Case 2.2: Temperament Quiz
- [ ] Complete all 15 quiz questions
- [ ] Verify slider interactions work
- [ ] Submit quiz
- [ ] **Expected**: Quiz scores saved, onboarding complete

#### Test Case 2.3: Skip Onboarding
- [ ] Click "Skip for now" on pet profile
- [ ] **Expected**: Redirected to home (limited functionality)

### 3. Match Discovery

#### Test Case 3.1: Browse Matches
- [ ] Navigate to `/discover`
- [ ] Verify 5+ match suggestions appear
- [ ] Check compatibility scores (should be >70%)
- [ ] **Expected**: ML-powered matches displayed with scores

#### Test Case 3.2: View Match Profile
- [ ] Click on a match card
- [ ] Verify pet details, owner info, temperament displayed
- [ ] Check compatibility breakdown
- [ ] **Expected**: Detailed profile with compatibility analysis

#### Test Case 3.3: Propose Meetup
- [ ] From match profile, click "Propose Meetup"
- [ ] Select location (map picker)
- [ ] Choose date/time
- [ ] Add optional message
- [ ] Submit
- [ ] **Expected**: Meetup proposal created, notification sent

### 4. Events System

#### Test Case 4.1: Browse Events
- [ ] Navigate to `/events`
- [ ] Filter by event type
- [ ] Toggle "Upcoming only"
- [ ] **Expected**: SF events displayed (Golden Gate Park, Dolores Park, etc.)

#### Test Case 4.2: RSVP to Event
- [ ] Click on event card
- [ ] Click "RSVP: Going"
- [ ] **Expected**: RSVP created, counter updated

#### Test Case 4.3: Check-in to Event
- [ ] Attend event (or simulate datetime)
- [ ] Click "Check In"
- [ ] **Expected**: Check-in recorded, **5 points awarded**

#### Test Case 4.4: Submit Event Feedback
- [ ] After event ends, submit feedback
- [ ] Rate: Vibe Score, Pet Density, Venue Quality
- [ ] Add tags and notes
- [ ] **Expected**: Feedback saved, **3 points awarded**

### 5. Service Discovery

#### Test Case 5.1: Browse Services
- [ ] Navigate to `/services`
- [ ] Filter by category (Grooming, Vet, Training, etc.)
- [ ] Apply distance filter
- [ ] **Expected**: SF service providers displayed

#### Test Case 5.2: Track Service Intent
- [ ] Click "Call" on a service
- [ ] **Expected**: Intent tracked with action='tap_call'
- [ ] Click "Book" on another service
- [ ] **Expected**: Intent tracked, n8n 24h follow-up triggered

#### Test Case 5.3: Service Conversion Follow-up
- [ ] Wait 24 hours (or manually trigger n8n)
- [ ] **Expected**: Push notification "Did you book [Service]?"

### 6. Gamification

#### Test Case 6.1: Earn Points (Multiple Ways)
- [ ] Create a post → **2 points**
- [ ] Like a post → **1 point**
- [ ] Check in to event → **5 points**
- [ ] Submit event feedback → **3 points**
- [ ] **Expected**: Points appear in profile, transactions logged

#### Test Case 6.2: View Leaderboard
- [ ] Navigate to `/leaderboard`
- [ ] **Expected**: Top 20 users by points displayed

#### Test Case 6.3: Badge Awards
- [ ] Complete first meetup
- [ ] **Expected**: "First Meetup" badge awarded
- [ ] Maintain 4-week streak
- [ ] **Expected**: "Streak Master" badge awarded

### 7. Push Notifications

#### Test Case 7.1: Enable Notifications
- [ ] Navigate to `/settings`
- [ ] Toggle "Enable Push Notifications"
- [ ] Grant browser permission
- [ ] **Expected**: Subscription stored, confirmation toast

#### Test Case 7.2: Receive Nudge Notification
- [ ] Get within 50m of compatible dog (or simulate)
- [ ] Wait for proximity check (every 5 minutes)
- [ ] **Expected**: Push notification with nudge

#### Test Case 7.3: Notification Click Action
- [ ] Click on push notification
- [ ] **Expected**: App opens to relevant page (/notifications, /events, etc.)

### 8. Proactive Nudges

#### Test Case 8.1: Proximity Nudge
- [ ] Be within 50m of compatible match for >5 min
- [ ] Ensure compatibility score ≥70%
- [ ] **Expected**: Nudge created, push notification sent
- [ ] Verify 24h cooldown (no duplicate nudges)

#### Test Case 8.2: Chat Activity Nudge
- [ ] Exchange 5+ messages in a match chat
- [ ] **Expected**: Meetup nudge suggested

#### Test Case 8.3: Accept Nudge
- [ ] Navigate to `/notifications`
- [ ] Click "Accept" on a nudge
- [ ] **Expected**: Redirected to chat/meetup flow

#### Test Case 8.4: Dismiss Nudge
- [ ] Click "Dismiss" on a nudge
- [ ] **Expected**: Nudge marked as dismissed, removed from list

### 9. Real-time Chat

#### Test Case 9.1: Send Message
- [ ] Navigate to `/messages`
- [ ] Select conversation
- [ ] Type message, press send
- [ ] **Expected**: Message appears instantly (WebSocket)

#### Test Case 9.2: Receive Message
- [ ] Have another user send you a message
- [ ] **Expected**: Message appears without refresh

#### Test Case 9.3: Image Upload
- [ ] Click image icon in chat
- [ ] Upload photo
- [ ] **Expected**: Image displayed in chat

---

## Performance Testing

### Page Load Times

| Page | Target | Measurement Method |
|------|--------|-------------------|
| Home (`/`) | <2s | Chrome DevTools Network tab |
| Match Discovery | <3s | Time to interactive |
| Event List | <2s | Lighthouse score >90 |
| Profile | <1.5s | First contentful paint |

### API Response Times

| Endpoint | Target | Test Tool |
|----------|--------|-----------|
| `GET /matches` | <500ms | Postman |
| `GET /events` | <300ms | curl with `-w "@curl-format.txt"` |
| `POST /nudges` | <200ms | Artillery.io |
| `POST /analytics/telemetry` | <100ms | Should not block UI |

### Stress Testing

```bash
# Test with Artillery
artillery quick --count 100 --num 10 http://localhost:4000/api/v1/health

# Expected:
# - 95th percentile latency < 500ms
# - 0% error rate
# - No memory leaks
```

---

## Security Testing

### Authentication

- [ ] Try accessing protected routes without JWT
  - **Expected**: 401 Unauthorized
- [ ] Try using expired JWT
  - **Expected**: 401 with "Token expired" message
- [ ] Verify CORS headers
  - **Expected**: Only allowed origins can make requests

### Input Validation

- [ ] Submit SQL injection in registration: `'; DROP TABLE users;--`
  - **Expected**: Validation error, no database changes
- [ ] XSS in chat message: `<script>alert('XSS')</script>`
  - **Expected**: Sanitized, displayed as plain text
- [ ] File upload: Try uploading `.exe` file as pet avatar
  - **Expected**: Only image files accepted

### Rate Limiting

- [ ] Make 150 requests to `/api/v1/auth/login` in 1 minute
  - **Expected**: 429 Too Many Requests after 100 requests
- [ ] Verify rate limit resets after TTL

---

## Analytics Validation

### Event Tracking

Verify these events are tracked in `/analytics/events`:

- [ ] `APP_OPEN` - On first load
- [ ] `SCREEN_VIEW` - On route change
- [ ] `MATCH_DISCOVERED` - When match appears
- [ ] `MEETUP_PROPOSED` - When proposing meetup
- [ ] `EVENT_CHECKIN` - When checking into event
- [ ] `POINTS_EARNED` - When points awarded

### North Star Metrics

Navigate to `/analytics/north-star` (admin only):

- [ ] **Meetup Conversion Rate**: Should show % of matches → confirmed meetups
- [ ] **7-Day Retention**: % of users returning within 7 days
- [ ] **Data Yield per User**: Avg labeled interactions per user
- [ ] **Service Intent Rate**: % of users tapping service CTAs

### Screen Views

```bash
# Query screen analytics
curl http://localhost:4000/api/v1/analytics/screens?timeframe=7d

# Expected: Top screens like:
# - discover (match discovery)
# - events (event list)
# - profile (user profiles)
```

---

## Known Issues

### Non-Critical (Will Fix Post-Launch)

1. **Image Upload Progress**: No progress bar during upload
   - **Workaround**: Show loading spinner

2. **Offline Mode**: Limited functionality when offline
   - **Workaround**: Service worker caching in place, but full offline mode pending

3. **Dark Mode**: Not yet implemented
   - **Status**: Planned for v1.1

### Critical (Must Fix Before Launch)

_None identified as of 2025-10-10_

---

## Reporting Bugs

### Bug Report Template

```markdown
**Title**: [Brief description]

**Severity**: Critical | High | Medium | Low

**Steps to Reproduce**:
1. Go to [page]
2. Click on [element]
3. ...

**Expected Behavior**: [What should happen]

**Actual Behavior**: [What actually happened]

**Environment**:
- Browser: [Chrome 120 / Safari 17 / etc.]
- OS: [macOS 14.0 / iOS 17.0 / etc.]
- Screen Size: [375x667 / 1920x1080]

**Screenshots/Logs**: [Attach if applicable]

**Additional Context**: [Any other relevant info]
```

### Submitting Reports

**Development Phase**:
- Create GitHub Issue: https://github.com/your-org/woof/issues

**Beta Phase**:
- Email: beta@woof.app
- Or use in-app feedback form (coming soon)

---

## Pre-Launch Checklist

### Infrastructure

- [ ] Production database configured
- [ ] Redis instance running
- [ ] n8n workflows imported and active
- [ ] Environment variables set correctly
- [ ] SSL certificates installed
- [ ] CDN configured for static assets

### Data

- [ ] SF seed data loaded
- [ ] 50+ real SF locations geocoded
- [ ] Service providers verified
- [ ] Events scheduled for next 30 days

### Monitoring

- [ ] Sentry error tracking configured
- [ ] Analytics dashboard accessible
- [ ] Uptime monitoring enabled
- [ ] Log aggregation set up

### Communication

- [ ] Beta tester invite list finalized
- [ ] Onboarding email template ready
- [ ] Support email monitored
- [ ] Feedback form live

---

## Success Criteria

### Week 1 (SF Beta)

- ✅ 50+ active users
- ✅ 100+ pet profiles created
- ✅ 20+ meetups proposed
- ✅ 5+ events attended
- ✅ <1% error rate
- ✅ 7-day retention >40%

### Week 4 (Expansion)

- ✅ 200+ active users
- ✅ 50+ confirmed meetups
- ✅ 15+ weekly events
- ✅ 100+ service intents tracked
- ✅ 7-day retention >50%

---

## Contact

**Product Lead**: [Your Name]
**Engineering**: [Engineering Lead]
**Support**: beta@woof.app

---

**Last Updated**: 2025-10-10
**Version**: 1.0.0
