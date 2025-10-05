# PetPath Beta MVP Implementation Roadmap

## 🎯 Mission
Build a beta that drives IRL meetups while collecting uniquely valuable datasets (pet compatibility, co-activity, meetup outcomes) that competitors ignore.

## 📊 North Star Metrics

| Metric | Target | Current | Tracking |
|--------|--------|---------|----------|
| **Meetup Conversion** | meetups_confirmed / unique_match_chats | - | ✅ UserInteraction table |
| **7D Retention** | % users returning within 7 days | - | ⏳ Need telemetry |
| **Data Yield per User** | labeled_interactions + verified_meetup_outcomes + co_activity_segments | - | ✅ MLTrainingData table |
| **Service Intent** | service_taps_to_book / MAU | - | ⏳ Need ServiceIntent tracking |

## 🏗️ System Architecture

### Data Layer (Enhanced Prisma Schema)

**New Tables Needed:**
```prisma
model CoActivitySegment {
  id            String   @id @default(uuid())
  userId        String   @map("user_id")
  petId         String   @map("pet_id")
  otherPetId    String?  @map("other_pet_id")
  startTime     DateTime @map("start_time")
  endTime       DateTime @map("end_time")
  distanceM     Float    @map("distance_m")
  gpsOverlapM   Float?   @map("gps_overlap_m")
  avgPace       Float?   @map("avg_pace")
  venueType     String?  @map("venue_type") // park, trail, urban, beach
  gpsTraceRef   String?  @map("gps_trace_ref") // S3/blob reference
  createdAt     DateTime @default(now()) @map("created_at")

  @@index([userId])
  @@index([petId])
  @@index([startTime])
  @@map("co_activity_segments")
}

model MeetupProposal {
  id              String    @id @default(uuid())
  matchId         String    @map("match_id")
  proposerId      String    @map("proposer_id")
  recipientId     String    @map("recipient_id")
  suggestedTime   DateTime  @map("suggested_time")
  suggestedVenue  Json      @map("suggested_venue") // {name, lat, lng, type}
  status          String    @default("pending") // pending, accepted, declined, completed
  occurredAt      DateTime? @map("occurred_at")
  rating          Int?      // 1-5
  feedbackTags    String[]  @map("feedback_tags") // energy_mismatch, size, temperament, great_match
  checklistOk     Boolean   @default(false) @map("checklist_ok")
  notes           String?
  createdAt       DateTime  @default(now()) @map("created_at")

  @@index([matchId])
  @@index([proposerId])
  @@index([recipientId])
  @@index([status])
  @@map("meetup_proposals")
}

model CommunityEvent {
  id              String   @id @default(uuid())
  title           String
  description     String?
  hostUserId      String   @map("host_user_id")
  venueType       String   @map("venue_type") // park, trail, beach, cafe
  lat             Float
  lng             Float
  venueName       String?  @map("venue_name")
  startTime       DateTime @map("start_time")
  endTime         DateTime @map("end_time")
  capacity        Int?
  rsvpCount       Int      @default(0) @map("rsvp_count")
  visibility      String   @default("PUBLIC") // PUBLIC, FRIENDS, PRIVATE
  recurring       Boolean  @default(false)
  postFeedbackScore Float? @map("post_feedback_score") // avg of all attendee ratings
  createdAt       DateTime @default(now()) @map("created_at")

  rsvps           EventRSVP[]
  feedback        EventFeedback[]

  @@index([hostUserId])
  @@index([startTime])
  @@index([lat, lng])
  @@map("community_events")
}

model EventRSVP {
  id          String    @id @default(uuid())
  eventId     String    @map("event_id")
  userId      String    @map("user_id")
  petId       String?   @map("pet_id")
  status      String    @default("YES") // YES, MAYBE, NO
  checkedInAt DateTime? @map("checked_in_at")
  createdAt   DateTime  @default(now()) @map("created_at")

  event CommunityEvent @relation(fields: [eventId], references: [id], onDelete: Cascade)

  @@unique([eventId, userId])
  @@index([eventId])
  @@index([userId])
  @@map("event_rsvps")
}

model EventFeedback {
  id           String   @id @default(uuid())
  eventId      String   @map("event_id")
  userId       String   @map("user_id")
  vibeScore    Int      @map("vibe_score") // 1-5
  petDensity   String?  @map("pet_density") // too_crowded, just_right, too_few
  surfaceType  String?  @map("surface_type")
  crowding     String?
  noiseLevel   String?  @map("noise_level")
  tags         String[]
  notes        String?
  createdAt    DateTime @default(now()) @map("created_at")

  event CommunityEvent @relation(fields: [eventId], references: [id], onDelete: Cascade)

  @@unique([eventId, userId])
  @@index([eventId])
  @@map("event_feedback")
}

model Business {
  id          String   @id @default(uuid())
  name        String
  type        String   // groomer, vet, trainer, store, park, restaurant, walker
  lat         Float
  lng         Float
  address     String?
  phone       String?
  website     String?
  rating      Float?
  hours       Json?    // {monday: "9-17", ...}
  petPolicies Json?    @map("pet_policies")
  partnered   Boolean  @default(false) // For rev-share partners
  createdAt   DateTime @default(now()) @map("created_at")

  serviceIntents ServiceIntent[]

  @@index([type])
  @@index([lat, lng])
  @@index([partnered])
  @@map("businesses")
}

model ServiceIntent {
  id                String    @id @default(uuid())
  userId            String    @map("user_id")
  businessId        String    @map("business_id")
  action            String    // view, tap_call, tap_book, tap_site
  conversionFollowup Boolean? @map("conversion_followup") // Did they actually book?
  createdAt         DateTime  @default(now()) @map("created_at")

  business Business @relation(fields: [businessId], references: [id])

  @@index([userId])
  @@index([businessId])
  @@index([action])
  @@index([createdAt])
  @@map("service_intents")
}

model Gamification {
  id           String   @id @default(uuid())
  userId       String   @unique @map("user_id")
  points       Int      @default(0)
  badges       String[] // first_match, first_meetup, first_event, pack_leader, etc
  weeklyStreak Int      @default(0) @map("weekly_streak")
  lastActive   DateTime @default(now()) @map("last_active")
  createdAt    DateTime @default(now()) @map("created_at")
  updatedAt    DateTime @updatedAt @map("updated_at")

  @@map("gamification")
}

model ProactiveNudge {
  id          String    @id @default(uuid())
  userId      String    @map("user_id")
  targetUserId String?  @map("target_user_id")
  type        String    // meetup, event, service, safety_check
  payload     Json      // Context data for the nudge
  accepted    Boolean?  // null = pending, true/false = responded
  sentAt      DateTime  @default(now()) @map("sent_at")
  respondedAt DateTime? @map("responded_at")

  @@index([userId])
  @@index([type])
  @@index([sentAt])
  @@map("proactive_nudges")
}

model SafetyVerification {
  id                String    @id @default(uuid())
  userId            String    @map("user_id")
  vaccineDocUrl     String?   @map("vaccine_doc_url")
  vaccineVerified   Boolean   @default(false) @map("vaccine_verified")
  idDocUrl          String?   @map("id_doc_url")
  idVerified        Boolean   @default(false) @map("id_verified")
  trustedBadge      Boolean   @default(false) @map("trusted_badge")
  verifiedAt        DateTime? @map("verified_at")
  createdAt         DateTime  @default(now()) @map("created_at")

  @@unique([userId])
  @@index([vaccineVerified])
  @@index([trustedBadge])
  @@map("safety_verifications")
}
```

## 🔥 Priority Feature Implementation

### Phase 1: Data Collection Foundation (Week 1-2)

**1. Enhanced Quiz System** ✅ DONE
- [x] 15-question compatibility quiz
- [ ] Add new questions: leash reactivity, noise tolerance, comfort with sizes
- [ ] Add schedule blocks with day-of-week granularity

**2. Meetup Proposal & Outcome Tracking** ⏳ HIGH PRIORITY
- [ ] Frontend: Meetup proposal card in chat
- [ ] "Suggest Midpoint" feature using Google Maps API
- [ ] Post-meetup feedback form (occurred?, rating, tags, notes)
- [ ] Meet-safe checklist (vaccines, leash, water, etc.)

**3. CoActivity Detection** ⏳ MEDIUM PRIORITY
- [ ] GPS tracking during activities (opt-in)
- [ ] Step counter integration (HealthKit/GoogleFit)
- [ ] Automatic "walked together" detection (proximity + time overlap)
- [ ] Store GPS traces with privacy controls

**4. Services Hub** ⏳ MEDIUM PRIORITY
- [ ] Business directory (groomers, vets, trainers, walkers)
- [ ] Service intent tracking (view, tap call/book/site)
- [ ] 24h follow-up: "Did you book?" → record conversion
- [ ] Partner badge for rev-share businesses

### Phase 2: Proactive Features (Week 3-4)

**5. Proactive Meetup Nudges** ⏳ HIGH PRIORITY
```typescript
// Nudge Engine Logic
interface NudgeSignals {
  proximity: boolean; // < 0.75 mi
  availability: boolean; // both free within 90 min
  energyState: boolean; // not fatigued
  safeVenue: boolean; // known park midpoint exists
}

// When >=3 signals → push notification
// "Buddy & Luna are nearby! Free for a walk at Riverside Park in 30 min?"
```

**6. Community Events** ⏳ HIGH PRIORITY
- [ ] Create event UI (title, time, venue, capacity)
- [ ] RSVP system with check-in
- [ ] Post-event feedback (vibe score, pet density, tags)
- [ ] Recurring event templates (weekly pack walks)
- [ ] Event discovery map

**7. Gamification v1** ⏳ MEDIUM PRIORITY
```typescript
const POINT_AWARDS = {
  onboarding_complete: 50,
  first_match: 25,
  first_meetup: 100,
  first_event: 75,
  first_review: 30,
  daily_login: 5,
  weekly_streak: 50,
};

const BADGES = [
  'first_match', 'first_meetup', 'pack_leader', 'explorer',
  'social_butterfly', 'event_host', 'trusted_owner'
];
```

### Phase 3: Safety & Privacy (Week 5)

**8. Safety Features** ⏳ HIGH PRIORITY
- [ ] Vaccine document upload + manual review
- [ ] Optional ID verification → "Trusted" badge
- [ ] Session-only location sharing (expires after meetup)
- [ ] Report/block functionality
- [ ] Meet-safe checklist before first meetup

**9. Privacy Controls** ⏳ MEDIUM PRIORITY
- [ ] Opt-in location sharing
- [ ] GPS trace anonymization (blur home/work addresses)
- [ ] Data export & deletion (GDPR compliance)
- [ ] Visibility settings (public, friends-only, private)

### Phase 4: Analytics & Experimentation (Week 6)

**10. Analytics Dashboard** ⏳ HIGH PRIORITY
```typescript
// North Star Metrics Dashboard
interface NSMetrics {
  meetupConversionRate: number; // meetups / match_chats
  retention7d: number; // % returning within 7d
  dataYieldPerUser: number; // avg labeled interactions
  serviceIntentRate: number; // service_taps / MAU
}

// Cohort Analysis
interface Cohort {
  signupWeek: string;
  size: number;
  completionRate: number;
  activeWeekly: number;
  meetupsAvg: number;
}
```

**11. A/B Testing Framework** ⏳ MEDIUM PRIORITY
```typescript
// Feature Flags + Experiments
const EXPERIMENTS = {
  explainability_chips: {
    hypothesis: 'Chips increase match→chat',
    variants: ['chips_on', 'chips_off'],
    successMetric: 'chat_started_rate',
    targetLift: 0.10, // +10%
  },
  midpoint_cta: {
    hypothesis: 'Auto-midpoint increases chat→meetup',
    variants: ['auto_midpoint', 'generic_cta'],
    successMetric: 'meetup_confirmed_rate',
    targetLift: 0.15, // +15%
  },
  neighborhood_packs: {
    hypothesis: 'Groups improve 7D retention',
    variants: ['groups_enabled', 'groups_disabled'],
    successMetric: 'retention_7d',
    targetLift: 0.08, // +8%
  },
};
```

## 🤖 n8n Automation Workflows

### Workflow 1: Proactive Meetup Nudge
```
Trigger: Location update OR availability window opens
→ Query compatible users within 0.75mi
→ Check availability overlap (90 min window)
→ Check energy state (not fatigued from recent activity)
→ Find safe venue midpoint
→ IF signals >= 3 → Send push nudge
→ Log ProactiveNudge record
→ Start 24h cooldown timer
```

### Workflow 2: Meetup Outcome Collector
```
Trigger: Meetup end_time + 30 min
→ Send in-app feedback card
→ "How was your playdate with Buddy?"
→ Rating (1-5), occurred (yes/no), tags (energy mismatch, great match, etc.)
→ Store in MeetupProposal.rating + feedbackTags
→ IF submitted → Award 30 points + update streak
```

### Workflow 3: Service Follow-Up
```
Trigger: ServiceIntent.action = 'tap_call' OR 'tap_book'
→ Wait 24 hours
→ Send push: "Did you book with [business]?"
→ Record response in ServiceIntent.conversionFollowup
→ IF yes → Notify business (partner rev-share)
```

### Workflow 4: Event Recap
```
Trigger: CommunityEvent.endTime + 1 hour
→ Send feedback survey to all checked-in attendees
→ Collect vibeScore, petDensity, tags
→ Calculate avg postFeedbackScore
→ Post community highlights ("12 pups had a blast at Riverside Pack Walk!")
→ Suggest follow-up event based on attendance
```

## 📱 UI/UX Enhancements

### Match Discovery
```
Current: Simple list
→ Enhanced: Card with explainability chips

[Profile Picture]
Buddy & Luna - 95% Match
🔋 Perfect energy match
📅 Both free Sat mornings
📍 0.3 mi away
🎾 Love fetch & hiking

[Like] [Skip] [Super Like]
```

### Chat → Meetup Flow
```
Chat Screen:
→ Floating action button: "Suggest Meetup"
→ Modal:
  - Auto-detected midpoint park (Google Maps)
  - Time picker (next 90 min slots highlighted)
  - Meet-safe checklist
  - [Send Proposal]
```

### Post-Meetup Feedback
```
Push Notification (30 min after scheduled end):
"How was your playdate with Buddy?"

Feedback Card:
- Did it happen? [Yes] [No] [Rescheduled]
- Rating: ⭐⭐⭐⭐⭐
- Tags: [Great energy match] [Size compatible] [Owner friendly]
- Notes: (optional text)
[Submit & Earn 30 Points!]
```

## 🎯 Beta Launch Targets (Per City)

| Metric | Target | Tracking |
|--------|--------|----------|
| Signups | 500 | Auth system |
| Onboarding Completion | 40% (200) | Quiz submissions |
| Weekly Active | 25% (125) | Telemetry events |
| Meetups (30 days) | 100+ | MeetupProposal.status = completed |
| Avg Meetup Rating | 4.0+/5.0 | MeetupProposal.rating |
| Service Taps | 50+ | ServiceIntent.count |

## 💰 Monetization Signals (Investor Metrics)

### Premium Trial Uptake
```typescript
const PREMIUM_FEATURES = [
  'priority_matching', // Show at top of discovery
  'advanced_filters', // Breed, size, temperament filters
  'see_who_liked', // Reveal mutual interest early
  'meetup_insurance', // $1M liability coverage bundle
  'unlimited_likes', // Free = 10/day, Premium = unlimited
];

// Track: % users who enable premium trial within 7 days
```

### Partner Rev-Share
```typescript
// Service Intent → Conversion Funnel
const CONVERSION_FUNNEL = {
  view_business: 100, // baseline
  tap_call: 15, // 15% click-to-call
  tap_book: 8, // 8% click-to-book
  confirmed_booking: 3, // 3% actual conversion (via follow-up)
};

// Partner Payment: $10-$25 per confirmed booking
// Target: 50 confirmed bookings/city/month = $500-$1250 MRR
```

## 📊 Investor Demo Flow (2 minutes)

```
0:00 - "Most pet apps show random dogs nearby with basic filters..."
0:15 - [Show competitor app] "That's it. No intelligence."

0:20 - "PetPath uses behavioral compatibility matching"
0:25 - [Show quiz] "15-question personality quiz"
0:40 - [Show match with chips] "Explainable 95% match score"

0:45 - [Show chat] "One-tap meetup suggestion with midpoint"
1:00 - [Show map] "Auto-finds safe park midway between owners"

1:10 - [Show post-meetup feedback] "Collect labeled outcome data"
1:20 - [Show analytics dashboard] "Track what matters: meetup conversion, retention, data yield"

1:30 - [Show services hub] "Local business partnerships"
1:40 - [Show gamification] "Engagement loops: points, badges, streaks"

1:50 - "We're not just matching pets. We're building the definitive dataset on pet compatibility and social behavior."
2:00 - [Show metrics] "100 meetups, 4.5⭐ avg rating, 30% 7D retention in beta"
```

## 🚧 Implementation Order

**Week 1-2: Core Data Flows**
1. Enhanced quiz with new questions
2. Meetup proposal system (frontend + backend)
3. Post-meetup feedback collection
4. Basic analytics dashboard

**Week 3-4: Proactive Features**
5. CoActivity GPS tracking (opt-in)
6. Proactive meetup nudge engine
7. Community events (create/join/RSVP)
8. Services hub + intent tracking

**Week 5: Safety & Polish**
9. Safety verifications (vaccine, ID)
10. Privacy controls (location opt-in, data export)
11. Gamification (points, badges, streaks)

**Week 6: Analytics & Launch Prep**
12. North star metrics dashboard
13. A/B testing framework
14. n8n automation workflows
15. Investor demo build

## 🎬 Next Actions

1. ✅ Update Prisma schema with new tables
2. ⏳ Implement meetup proposal UI components
3. ⏳ Build services hub directory
4. ⏳ Create analytics dashboard
5. ⏳ Set up n8n workflows

---

**Status**: Database foundation ✅ | Backend 30% | Frontend 40% | Analytics 0% | Automation 0%
**Target Launch**: 6 weeks from now
**Priority**: Meetup conversion data collection (highest ROI for ML training)
