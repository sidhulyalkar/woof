# 🚀 Woof Beta Readiness Roadmap

**Current Status**: ✅ Core Features Complete | 🔄 Beta Enhancement Phase
**Target**: San Francisco Closed Beta Launch
**Last Updated**: October 9, 2025

---

## 📊 Overview

This roadmap outlines the remaining tasks to prepare Woof for a successful closed beta launch in San Francisco. All foundational infrastructure (testing, CI/CD, security, core features) is complete. This phase focuses on user engagement automation, polish, and monitoring.

---

## 🎯 High Priority Tasks (Launch Blockers)

### Task 1: Proactive Meetup Nudge Engine ⚡

**Priority**: HIGH | **Estimated Time**: 2-3 days | **Status**: 🔄 Pending

#### Description
Build automated nudge system to proactively suggest meetups based on dynamic conditions (proximity, chat activity, availability).

#### Implementation Plan

**Backend Components**:

1. **Create Nudge Service** (`apps/api/src/nudges/nudges.service.ts`)
   ```typescript
   @Injectable()
   export class NudgesService {
     // Condition 1: Location proximity detection
     async checkProximityNudges(): Promise<void>

     // Condition 2: Chat activity monitoring
     async checkChatActivityNudges(): Promise<void>

     // Condition 3: Cooldown enforcement
     async canSendNudge(userId1: string, userId2: string): Promise<boolean>

     // Create and dispatch nudge
     async createNudge(data: CreateNudgeDto): Promise<ProactiveNudge>
   }
   ```

2. **Add Cron Jobs** (NestJS Schedule)
   ```typescript
   @Cron('*/5 * * * *') // Every 5 minutes
   async checkProximityConditions() {
     // Query recent CoActivitySegments
     // Find users within 50m for >5 minutes
     // Check compatibility scores
     // Create proximity nudges
   }
   ```

3. **Create DTOs** (`nudges/dto/create-nudge.dto.ts`)
   ```typescript
   export class CreateNudgeDto {
     userId: string;
     type: 'meetup' | 'service' | 'event';
     context: {
       targetUserId?: string;
       location?: { lat: number; lng: number };
       reason: 'proximity' | 'chat_activity' | 'mutual_availability';
     };
   }
   ```

4. **Database Updates** (Prisma schema already has ProactiveNudge model)
   - Add indexes for performance:
     ```prisma
     @@index([userId, type, createdAt])
     @@index([dismissed, createdAt])
     ```

**Frontend Components**:

1. **Nudge Notification Component**
   ```tsx
   // apps/web/src/components/notifications/nudge-notification.tsx
   export function NudgeNotification({ nudge }: { nudge: ProactiveNudge }) {
     // Display in-app notification
     // Action buttons: "Let's meet!" or "Dismiss"
   }
   ```

2. **Notifications Center**
   ```tsx
   // apps/web/src/app/notifications/page.tsx
   // List all active nudges
   // Mark as read/dismissed
   ```

**Acceptance Criteria**:
- ✅ Users within 50m for >5 min get proximity nudge
- ✅ Chat activity (5+ messages) triggers meetup suggestion
- ✅ Cooldown prevents spam (24h between same-pair nudges)
- ✅ Nudges appear in notification center
- ✅ Users can accept or dismiss nudges

**Files to Create**:
- `apps/api/src/nudges/nudges.module.ts`
- `apps/api/src/nudges/nudges.service.ts`
- `apps/api/src/nudges/nudges.controller.ts`
- `apps/api/src/nudges/dto/create-nudge.dto.ts`
- `apps/web/src/app/notifications/page.tsx`
- `apps/web/src/components/notifications/nudge-notification.tsx`

---

### Task 2: Push Notification Service 📱

**Priority**: HIGH | **Estimated Time**: 1-2 days | **Status**: 🔄 Pending

#### Description
Enable real-time push notifications for critical events (nudges, achievements, reminders) to re-engage users when app is closed.

#### Implementation Plan

**Choose Platform**:
- **Web PWA**: Web Push API with VAPID keys
- **Future Mobile**: Expo Push or Firebase Cloud Messaging

**Backend Setup**:

1. **Install Dependencies**
   ```bash
   pnpm --filter @woof/api add web-push
   ```

2. **Create Notifications Module**
   ```typescript
   // apps/api/src/notifications/notifications.service.ts
   @Injectable()
   export class NotificationsService {
     async sendPushNotification(userId: string, data: PushPayload)
     async subscribePushNotification(userId: string, subscription: PushSubscription)
     async unsubscribePushNotification(userId: string, endpoint: string)
   }
   ```

3. **Generate VAPID Keys**
   ```bash
   npx web-push generate-vapid-keys
   # Add to .env: VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY
   ```

4. **Add Endpoints**
   ```typescript
   @Post('subscribe')
   async subscribe(@Body() data: { userId: string; subscription: PushSubscription })

   @Post('send')
   async send(@Body() data: { userId: string; title: string; body: string; data?: any })
   ```

**Frontend Setup**:

1. **Service Worker** (already exists, extend it)
   ```javascript
   // apps/web/public/sw.js
   self.addEventListener('push', function(event) {
     const data = event.data.json();
     self.registration.showNotification(data.title, {
       body: data.body,
       icon: '/icon-192.png',
       badge: '/badge-72.png',
       data: data.data
     });
   });

   self.addEventListener('notificationclick', function(event) {
     event.notification.close();
     event.waitUntil(
       clients.openWindow(event.notification.data.url || '/')
     );
   });
   ```

2. **Push Subscription Component**
   ```tsx
   // apps/web/src/hooks/use-push-notifications.ts
   export function usePushNotifications() {
     const subscribe = async () => {
       const registration = await navigator.serviceWorker.ready;
       const subscription = await registration.pushManager.subscribe({
         userVisibleOnly: true,
         applicationServerKey: urlBase64ToUint8Array(VAPID_PUBLIC_KEY)
       });

       await apiClient.post('/notifications/subscribe', {
         userId: currentUser.id,
         subscription
       });
     };

     return { subscribe };
   }
   ```

**Integration Points**:
- Nudge creation → Push notification
- Gamification achievement → Push notification
- Event reminders → Push notification
- Service follow-ups → Push notification

**Acceptance Criteria**:
- ✅ Users can enable push notifications
- ✅ Nudges trigger push notifications
- ✅ Clicking notification opens app at relevant screen
- ✅ System handles 100+ notifications/day
- ✅ Users can disable notifications in settings

**Files to Create**:
- `apps/api/src/notifications/notifications.module.ts`
- `apps/api/src/notifications/notifications.service.ts`
- `apps/api/src/notifications/notifications.controller.ts`
- `apps/web/src/hooks/use-push-notifications.ts`
- `apps/web/src/components/settings/notification-settings.tsx`

---

### Task 3: n8n Automation Workflows 🤖

**Priority**: HIGH | **Estimated Time**: 2 days | **Status**: 🔄 Pending

#### Description
Create automated workflows for follow-ups, reminders, and user re-engagement using n8n.

#### Workflows to Build

**1. Service Booking 24h Follow-up**

```
Trigger: ServiceIntent created with action='tap_book'
  ↓
Wait 24 hours
  ↓
Check if conversion recorded (conversionFollowup=true)
  ↓ (if false)
Send notification: "Did you book [ServiceName]?"
  ↓
Wait for response (3 days max)
  ↓
Update ServiceIntent.conversionFollowup based on response
```

**2. Meetup Feedback Reminder**

```
Trigger: Meetup.datetime passes
  ↓
Wait 2 hours after meetup time
  ↓
Check if meetup.status still 'pending'
  ↓ (if yes)
Send notification to both users: "Did your meetup with [User] happen?"
  ↓
Include deep link to feedback form
  ↓
When feedback submitted → Update meetup.status to 'completed'
```

**3. Event Reminder & Follow-up**

```
Trigger: Event.datetime - 3 hours
  ↓
Get all RSVPs with response='yes'
  ↓
Send reminder notification to each attendee
  ↓
After event ends (datetime + duration)
  ↓
Check who hasn't submitted feedback
  ↓
Send feedback request notification
```

**4. Fitness Goal Achievement**

```
Trigger: Daily cron (9 PM)
  ↓
Query users with weekly goals
  ↓
Calculate progress from Activity logs
  ↓ (if goal met)
Award points via API: POST /gamification/points
  ↓
Send congratulatory push notification
```

**Implementation**:

1. **Set up n8n in docker-compose.yml** (already done)
2. **Create webhook endpoints** in NestJS to trigger workflows
3. **Build each workflow** in n8n UI
4. **Test with staging data**
5. **Monitor execution logs**

**Acceptance Criteria**:
- ✅ Service booking follow-ups sent 24h after tap_book
- ✅ Meetup participants reminded to confirm/rate
- ✅ Event attendees get day-of reminder
- ✅ Goal achievements trigger automatic rewards
- ✅ All workflows log actions for audit

**Documentation**:
- Create `n8n-workflows/README.md` with setup instructions
- Export workflows as JSON for version control

---

## 🎨 Medium Priority Tasks (Polish & Engagement)

### Task 4: Gamification Trigger Integration

**Priority**: MEDIUM | **Estimated Time**: 1 day | **Status**: 🔄 Pending

#### Auto-Award Points

Add point triggers in controllers:

```typescript
// In meetups.controller.ts - submitFeedback()
if (feedbackDto.occurred) {
  await this.gamificationService.awardPoints({
    userId: req.user.id,
    points: 10,
    reason: 'meetup_completed'
  });
}

// In events.controller.ts - checkIn()
await this.gamificationService.awardPoints({
  userId: req.user.id,
  points: 5,
  reason: 'event_attended'
});

// In social.controller.ts - likePost()
await this.gamificationService.awardPoints({
  userId: req.user.id,
  points: 1,
  reason: 'post_liked'
});
```

#### Auto-Award Badges

```typescript
// After first meetup completion
const meetupCount = await this.prisma.meetup.count({
  where: {
    OR: [{ proposerId: userId }, { proposeeId: userId }],
    status: 'completed'
  }
});

if (meetupCount === 1) {
  await this.gamificationService.awardBadge({
    userId,
    badge: 'first_meetup'
  });
}
```

#### Frontend Reward Feedback

```tsx
// Show celebration modal after earning badge/points
<ConfettiModal
  title="Achievement Unlocked!"
  badge="First Meetup"
  points={10}
  onClose={() => router.push('/profile')}
/>
```

**Files to Modify**:
- `apps/api/src/meetups/meetups.controller.ts`
- `apps/api/src/events/events.controller.ts`
- `apps/api/src/social/social.controller.ts`
- `apps/web/src/components/gamification/achievement-modal.tsx`

---

### Task 5: UI/UX Polish & Consistency

**Priority**: MEDIUM | **Estimated Time**: 2 days | **Status**: 🔄 Pending

#### Responsive Design Audit

**Test Viewports**:
- Desktop: 1920x1080
- Tablet: 768x1024
- Mobile: 375x667 (iPhone SE)
- Mobile Large: 414x896 (iPhone 11)

**Pages to Audit**:
1. Match Discovery
2. Event List/Detail
3. Chat Interface
4. Profile Pages
5. Onboarding Flow

**Fixes Needed**:
```tsx
// Example: Make event cards responsive
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {events.map(event => <EventCard key={event.id} event={event} />)}
</div>
```

#### Theme Consistency

**Color Audit**:
- Replace any hard-coded colors with Tailwind classes
- Verify all backgrounds use theme colors
- Check text contrast ratios (WCAG AA compliance)

```tsx
// Before
<div style={{ backgroundColor: '#1a1a1a' }}>

// After
<div className="bg-surface">
```

#### Interactive States

Ensure all interactive elements have:
- Hover states (`:hover`)
- Active states (`:active`)
- Focus states (`:focus-visible`)
- Disabled states (`disabled:`)

```tsx
<Button
  className="
    bg-primary hover:bg-primary/90
    active:scale-95
    focus-visible:ring-2 focus-visible:ring-accent
    disabled:opacity-50 disabled:cursor-not-allowed
  "
>
  Submit
</Button>
```

**Acceptance Criteria**:
- ✅ All pages look perfect on mobile, tablet, desktop
- ✅ No color inconsistencies across app
- ✅ All interactive states clearly visible
- ✅ No UI overflow or misalignment
- ✅ Passes internal design review

---

### Task 6: Analytics & Telemetry

**Priority**: MEDIUM | **Estimated Time**: 1 day | **Status**: 🔄 Pending

#### Event Tracking Implementation

**Frontend Events to Track**:

```typescript
// apps/web/src/lib/analytics.ts
export const trackEvent = async (event: string, properties?: Record<string, any>) => {
  await apiClient.post('/analytics/telemetry', {
    source: 'WEB',
    event,
    metadata: properties
  });
};

// Usage across app
trackEvent('APP_OPEN', { timestamp: new Date() });
trackEvent('SCREEN_VIEW', { screen: 'match_discovery' });
trackEvent('MEETUP_PROPOSED', { targetUserId });
trackEvent('PROFILE_COMPLETED', { hasQuiz: true });
```

**Backend Telemetry Endpoint**:

```typescript
@Post('telemetry')
async recordTelemetry(@Body() data: TelemetryDto) {
  return this.prisma.telemetry.create({ data });
}
```

**Retention Calculation**:

```typescript
// apps/api/src/analytics/analytics.service.ts
async calculate7DayRetention(): Promise<number> {
  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

  const activeUsers = await this.prisma.telemetry.groupBy({
    by: ['userId'],
    where: {
      event: 'APP_OPEN',
      createdAt: { gte: sevenDaysAgo }
    }
  });

  const totalUsers = await this.prisma.user.count();
  return (activeUsers.length / totalUsers) * 100;
}
```

**Acceptance Criteria**:
- ✅ All key user actions tracked
- ✅ 7-day retention metric shows real data
- ✅ No performance impact from tracking
- ✅ Analytics dashboard populated with insights

---

## 🧪 Launch Preparation (Final Phase)

### Task 7: Beta Testing & Hardening

**Priority**: MEDIUM | **Estimated Time**: 3 days | **Status**: 🔄 Pending

#### Testing Checklist

**Functional Testing**:
- [ ] Complete user registration flow
- [ ] Pet profile creation with quiz
- [ ] Match discovery and compatibility
- [ ] Propose meetup (accept/decline/feedback)
- [ ] Create and attend event
- [ ] Service discovery and intent tracking
- [ ] Real-time chat messaging
- [ ] Activity logging with photos
- [ ] Gamification (points, badges, streaks)

**Cross-Browser Testing**:
- [ ] Chrome (desktop + mobile)
- [ ] Safari (desktop + iOS)
- [ ] Firefox
- [ ] Edge

**Performance Testing**:
- [ ] Page load times <2s
- [ ] API response times <500ms
- [ ] No memory leaks (leave app open 1h)
- [ ] Image optimization working
- [ ] Database query performance

**Security Testing**:
- [ ] Rate limiting functional
- [ ] JWT expiration working
- [ ] CORS configured correctly
- [ ] File upload validation
- [ ] XSS prevention verified

**Monitoring Setup**:
- [ ] Sentry capturing errors
- [ ] Vercel Analytics tracking
- [ ] Custom metrics dashboard
- [ ] Uptime monitoring (UptimeRobot)

**Deployment Checklist**:
- [ ] Environment variables set
- [ ] Database migrations run
- [ ] Seed data loaded
- [ ] SSL certificates valid
- [ ] CDN configured
- [ ] Backup strategy in place

**Beta User Onboarding**:
- [ ] Terms of Service published
- [ ] Privacy Policy published
- [ ] Feedback form created
- [ ] Support email configured
- [ ] Beta welcome email template

---

## 📅 Estimated Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| **Week 1** | 5 days | Tasks 1-3 (High Priority) |
| **Week 2** | 3 days | Tasks 4-6 (Medium Priority) |
| **Week 3** | 3 days | Task 7 (Testing & Hardening) |
| **Total** | ~11 days | Ready for Beta Launch |

---

## 🎯 Success Metrics (Post-Launch)

### Week 1 Goals:
- 50+ registered users
- 10+ successful meetups
- 5+ user-created events
- 100+ activities logged
- <5 critical bugs reported

### Week 4 Goals:
- 30% 7-day retention
- 50+ repeat meetups
- 10+ service conversions
- 200+ daily active users
- 4.0+ average app rating

---

## 📝 Notes

### Current Status Summary:
- ✅ **Complete**: Core features, testing, CI/CD, security, seed data
- 🔄 **In Progress**: Beta enhancements (this roadmap)
- ⏳ **Pending**: Tasks 1-7 above

### Priority Adjustments:
- Tasks 1-3 are **launch blockers** - must complete before beta
- Tasks 4-6 enhance engagement - complete before/during early beta
- Task 7 is ongoing throughout beta period

### Resources Needed:
- Development: 1 full-stack developer (~2 weeks)
- Design: Minor UI tweaks and asset review (2-3 days)
- QA: Beta testing coordination (ongoing)

---

**Last Updated**: October 9, 2025
**Next Review**: After Task 3 completion
**Owner**: Development Team
