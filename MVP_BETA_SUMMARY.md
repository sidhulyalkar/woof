# PetPath MVP Beta - Complete Feature Set

## 🎯 Project Vision
**PetPath** is a dog owner social network focused on driving **IRL (in-real-life) meetups** while collecting uniquely valuable datasets that competitors ignore. The app positions itself for superior ML models and product insights through systematic outcome labeling and conversion tracking.

## 📊 North Star Metrics

1. **Meetup Conversion Rate** = confirmed meetups / unique match chats
2. **7-Day Retention** = % users returning within 7 days
3. **Data Yield per User** = labeled interactions + verified outcomes
4. **Service Intent Rate** = service taps to book / MAU

---

## 🏗️ Technical Architecture

### Stack
- **Backend**: NestJS + Prisma + PostgreSQL
- **Frontend**: Next.js 14 + React + TypeScript + Tailwind CSS + shadcn/ui
- **Authentication**: JWT with refresh tokens
- **API**: RESTful with Swagger documentation
- **Automation**: n8n workflows (planned)

### Database
- 14 new beta MVP tables in Prisma schema
- Type-safe queries throughout
- Optimized for ML training data collection

---

## ✅ Backend Modules (Complete)

### 1. **Meetup Proposals**
`apps/api/src/meetup-proposals/`

**Purpose**: Full lifecycle management for 1-on-1 meetup proposals

**Features**:
- Create proposals with time/location/activity type
- Accept/reject/confirm workflow
- Feedback collection after meetups
- Did-it-happen tracking + rating (1-5) + tags
- Statistics per user (pending/accepted/completed/avg rating)

**Key Endpoints**:
- `POST /meetup-proposals` - Create proposal
- `GET /meetup-proposals` - Get sent/received proposals
- `PUT /meetup-proposals/:id/status` - Update status
- `POST /meetup-proposals/:id/complete` - Submit feedback
- `GET /meetup-proposals/stats` - Get user statistics

**Data Collection**:
- ✅ Outcome labeling (did it happen?)
- ✅ Quality ratings
- ✅ Behavioral tags for ML training

---

### 2. **Services Hub**
`apps/api/src/services/`

**Purpose**: Business directory with conversion intent tracking

**Features**:
- CRUD for pet businesses (groomers, vets, boarding, etc.)
- Service intent tracking (view, tap_call, tap_directions, tap_website, tap_book)
- 24-hour follow-up for conversion confirmation
- Conversion statistics (tap_book → actual booking rate)

**Key Endpoints**:
- `GET /services/businesses` - List all businesses
- `POST /services/intents` - Track user action
- `GET /services/stats/conversion` - Get conversion metrics

**Data Collection**:
- ✅ Service discovery patterns
- ✅ Monetization signals (tap-to-book actions)
- ✅ Conversion tracking (competitors don't do this!)

---

### 3. **Community Events**
`apps/api/src/events/`

**Purpose**: Group events with RSVP and multi-dimensional feedback

**Features**:
- Create events (group walks, park meetups, training sessions)
- RSVP system (going/maybe/not_going) with capacity management
- Multi-dimensional feedback: vibeScore, petDensity, venueQuality (1-5 each)
- Feedback tags for qualitative insights
- Average calculations for event quality scoring

**Key Endpoints**:
- `POST /events` - Create event
- `GET /events` - List upcoming/past events
- `POST /events/:id/rsvp` - RSVP to event
- `POST /events/:id/feedback` - Submit feedback
- `GET /events/:id/feedback` - Get aggregated feedback

**Data Collection**:
- ✅ Event attendance predictions
- ✅ Venue quality data for recommendations
- ✅ Social dynamics insights (pet density, vibe)

---

### 4. **Gamification**
`apps/api/src/gamification/`

**Purpose**: Points, badges, and streaks to drive engagement

**Features**:
- Point transactions with reason tracking
- 8 badge types (first_match, streak_master, verified_owner, etc.)
- Weekly streak calculation with auto-reset
- Leaderboard (top users by points)
- Summary endpoint for user's gamification status

**Key Endpoints**:
- `POST /gamification/points` - Award points
- `GET /gamification/points/:userId` - Get user points
- `POST /gamification/badges` - Award badge
- `POST /gamification/streaks` - Update streak
- `GET /gamification/leaderboard` - Top users
- `GET /gamification/me/summary` - User's gamification stats

**Data Collection**:
- ✅ Engagement signals
- ✅ Retention drivers
- ✅ User commitment levels

---

### 5. **Safety Verification**
`apps/api/src/verification/`

**Purpose**: Document upload and admin review for trust & safety

**Features**:
- Multi-document type support (vaccination, license, ID, vet certificate)
- Pending/approved/rejected workflow
- Admin review with notes
- Auto-update user's isVerified flag on approval
- Statistics dashboard

**Key Endpoints**:
- `POST /verification/upload` - Upload document
- `GET /verification/me` - Get my verifications
- `GET /verification/pending` - Admin: pending queue
- `PATCH /verification/:id` - Admin: update status
- `GET /verification/stats` - Admin: statistics

**Data Collection**:
- ✅ Trust & safety signals
- ✅ User verification rates
- ✅ Community quality metrics

---

### 6. **Co-Activity Tracking**
`apps/api/src/co-activity/`

**Purpose**: GPS overlap detection for serendipitous discovery

**Features**:
- Location ping tracking with timestamps
- Haversine distance calculation (50m proximity threshold)
- Overlap detection (within 30min time window)
- Potential match finding based on co-location patterns
- Statistics per user

**Key Endpoints**:
- `POST /co-activity/track` - Track location
- `GET /co-activity/me/locations` - My location history
- `GET /co-activity/overlaps/:userId` - Detect overlaps with user
- `GET /co-activity/me/matches` - Find potential matches
- `GET /co-activity/me/stats` - My co-activity stats

**Data Collection**:
- ✅ Real-world movement patterns
- ✅ Neighborhood density heatmaps
- ✅ Serendipitous encounter data

---

### 7. **Analytics**
`apps/api/src/analytics/`

**Purpose**: North star metrics calculation and tracking

**Features**:
- North star metrics with trend calculations
- Meetup funnel visualization
- Service intent conversion tracking
- Event feedback aggregation
- User engagement metrics
- Timeframe support (7d/30d/90d)

**Key Endpoints**:
- `GET /analytics/north-star?timeframe=30d` - Get north star metrics
- `GET /analytics/details?timeframe=30d` - Detailed breakdown

**Metrics Calculated**:
- Meetup conversion rate (+ trend)
- 7D retention rate (+ trend)
- Data yield per user (+ trend)
- Service intent rate (+ trend)
- Total/active users
- Meetup funnel (matches → confirmed → completed)
- Service conversions (tap_book → actual booking)
- Event feedback quality (avg scores)

---

## 🎨 Frontend Components (Complete)

### 1. **MatchDiscovery Screen**
`apps/web/src/components/matches/MatchDiscoveryScreen.tsx`

**Features**:
- Tinder-style swipe cards
- Compatibility score display (e.g., "85% Match")
- Explainability chips ("Both love hiking", "Similar energy levels")
- Distance display (e.g., "2.3km away")
- Last active status
- Like/Skip/Super Like actions with animations
- Interaction tracking for ML training

**UX Highlights**:
- Swipe animations (left = skip, right = like)
- Auto-refetch when running low on matches
- Empty state with friendly messaging

---

### 2. **MeetupProposal Screen**
`apps/web/src/components/meetups/MeetupProposalScreen.tsx`

**Features**:
- Create proposal form (time, location, activity, message)
- Pending/confirmed/past meetup views
- Accept/reject actions
- Feedback collection modal
  - Did-it-happen toggle
  - Rating slider (1-5)
  - Quick tags ("Great conversation", "Pets got along")
  - Comments field
- Full detail modal for each proposal

**UX Highlights**:
- Status badges (pending/confirmed/completed)
- RSVP-style UI for proposals
- Contextual CTAs based on proposal status

---

### 3. **EnhancedChat Screen**
`apps/web/src/components/chat/EnhancedChatScreen.tsx`

**Features**:
- Match list with compatibility scores
- Real-time message polling (3s interval)
- **Proactive meetup CTA** after 5+ messages exchanged
- "Propose Meetup" button in header
- Unread message badges
- Message bubbles with timestamps
- Deep integration with MeetupProposalScreen

**UX Highlights**:
- Auto-triggered CTA banner when threshold met
- One-tap meetup proposal from chat
- Persistent "Meetup" header button

---

### 4. **Events Screen**
`apps/web/src/components/events/EventsScreen.tsx`

**Features**:
- Create event dialog (time, location, capacity, tags)
- Upcoming/past tabs
- RSVP buttons (going/maybe/not_going)
- Capacity warnings ("5 spots left!")
- Event detail modal
- Feedback collection for past events
  - Vibe score slider (1-5)
  - Pet density slider (1-5)
  - Venue quality slider (1-5)
  - Quick tags
- Aggregated feedback display

**UX Highlights**:
- Calendar-style date badges
- Organizer info prominently displayed
- "Leave Feedback" CTA for attended events

---

### 5. **ServicesHub Screen**
`apps/web/src/components/services/ServicesHubScreen.tsx`

**Features**:
- Business directory with search
- Filter by type (groomer, vet, boarding, all)
- Business cards with photos
- Action buttons:
  - Call (tap_call)
  - Directions (tap_directions)
  - Website (tap_website)
  - **Book** (tap_book) - triggers conversion tracking
- Intent tracking on all actions

**UX Highlights**:
- Toast on "Book" action: "We'll follow up in 24h to see if you booked"
- Service tags displayed (e.g., "nail trim", "bath", "daycare")
- Distance display for nearby businesses

---

### 6. **Verification Screen**
`apps/web/src/components/verification/VerificationScreen.tsx`

**Features**:
- Document upload form
  - File input (JPG/PNG/PDF, max 10MB)
  - Document type selector
  - Notes field
- Upload validation
- Uploaded documents list with status badges
- Review notes display (for rejected docs)
- Shield icon for verified users

**UX Highlights**:
- Visual feedback during upload
- Clear status indicators (pending/approved/rejected)
- Admin review notes shown to user

---

### 7. **GamificationWidget**
`apps/web/src/components/profile/GamificationWidget.tsx`

**Features**:
- Points/badges/streaks display
- Badge icons with labels
- Streak fire animation
- Encouragement messages
- Empty state for new users

**UX Highlights**:
- Gradient card design
- Visual hierarchy (large numbers, small labels)
- Contextual messaging based on progress

---

### 8. **AnalyticsDashboard**
`apps/web/src/components/analytics/AnalyticsDashboard.tsx`

**Features**:
- North star metrics cards with trends
- Timeframe filters (7d/30d/90d)
- Tabbed views:
  - Meetups (funnel visualization)
  - Services (conversion tracking)
  - Events (feedback quality)
  - Engagement (user activity)
- Color-coded metric cards
- Trend indicators (up/down arrows)
- Funnel progress bars

**UX Highlights**:
- Executive dashboard aesthetic
- Data-driven insights at a glance
- Drill-down tabs for detailed analysis

---

## 🔄 Data Collection Strategy

### What Makes PetPath Unique

**Competitors** (Rover, BarkHappy, etc.) collect:
- Basic profile data
- Service bookings
- Simple ratings

**PetPath** collects:
- ✅ **Meetup outcomes** - Did the proposed meetup actually happen?
- ✅ **Quality labels** - How was the meetup? (rating + tags)
- ✅ **Service conversions** - Did they book after clicking "Book"?
- ✅ **Event feedback** - Multi-dimensional ratings for ML training
- ✅ **Co-activity patterns** - Real-world GPS overlap data
- ✅ **Engagement signals** - Streaks, points, badge unlocks

### ML Training Data Pipeline

1. **Meetup Proposals** → Outcome labels (success/failure)
2. **Service Intents** → Conversion labels (booked/didn't book)
3. **Event Feedback** → Quality scores (vibe/density/venue)
4. **Match Interactions** → Preference signals (like/skip/super_like)
5. **Co-Activity** → Movement patterns and neighborhood preferences

This data enables:
- **Better matching algorithms** (learn from successful meetups)
- **Predictive models** (which proposals will convert?)
- **Venue recommendations** (which parks/cafés work best?)
- **Monetization** (service conversion prediction)

---

## 📈 Key Features for IRL Meetups

1. **Proactive Nudges**
   - Chat CTA after 5+ messages
   - Proximity alerts (nearby matches)
   - Energy level matching
   - Time availability signals

2. **Friction Reduction**
   - One-tap proposal from chat
   - Pre-filled location suggestions
   - Activity type quick-select
   - Calendar integration (planned)

3. **Outcome Tracking**
   - Did-it-happen confirmation
   - Quality ratings
   - Behavioral tags
   - Follow-up prompts

4. **Trust & Safety**
   - Verified owner badges
   - Document verification
   - Rating history
   - Report functionality (planned)

---

## 🚀 Implementation Status

### ✅ Completed
- [x] All 7 backend modules
- [x] All 8 frontend screens
- [x] North star metrics calculation
- [x] Analytics dashboard
- [x] Data collection infrastructure
- [x] API documentation (Swagger)
- [x] Type-safe database schema
- [x] JWT authentication
- [x] Form validation

### 🔄 In Progress
- [ ] n8n automation workflows
- [ ] Proactive nudge engine
- [ ] Push notifications
- [ ] Calendar integration

### ⏳ Planned
- [ ] Seed data for testing
- [ ] E2E test coverage
- [ ] Performance optimization
- [ ] Error monitoring (Sentry)
- [ ] Analytics event tracking

---

## 🎯 Next Steps

1. **Seed Data Creation**
   - Generate sample users
   - Create mock meetup proposals
   - Add sample businesses
   - Schedule test events

2. **Testing**
   - E2E user flows
   - API endpoint validation
   - Edge case handling
   - Mobile responsiveness

3. **Deployment**
   - Set up staging environment
   - Configure production database
   - Set up CI/CD pipeline
   - Deploy to Vercel + Railway

4. **Launch Prep**
   - Beta user recruitment
   - Onboarding flow optimization
   - Performance monitoring setup
   - Support infrastructure

---

## 📝 API Documentation

API docs available at: `http://localhost:4000/docs` (Swagger UI)

### Key Endpoint Groups
- `/auth` - Authentication (register, login, me)
- `/users` - User management
- `/pets` - Pet profiles
- `/meetup-proposals` - Meetup lifecycle
- `/services` - Business directory + intent tracking
- `/events` - Community events + RSVP + feedback
- `/gamification` - Points/badges/streaks
- `/verification` - Document upload + review
- `/co-activity` - GPS tracking + overlap detection
- `/analytics` - North star metrics + detailed breakdown

---

## 🏆 Competitive Advantages

1. **Data Moat**: Unique conversion tracking + outcome labels
2. **IRL Focus**: Every feature drives offline meetups
3. **ML Readiness**: Systematic data collection for superior algorithms
4. **Gamification**: Streaks + badges drive retention
5. **Safety**: Document verification + trust signals
6. **Monetization**: Service intent tracking for B2B opportunities

---

## 💡 Product Philosophy

**"Collect data competitors ignore, optimize for IRL meetups"**

- Every interaction labeled
- Every outcome tracked
- Every conversion measured
- Every behavior signaled

This creates a **compounding data advantage** that competitors cannot replicate without rebuilding their entire product.

---

## 🛠️ Development Commands

```bash
# Start API server
cd apps/api && pnpm dev

# Start frontend
cd apps/web && pnpm dev

# Database migrations
cd packages/database && npx prisma migrate dev

# Generate Prisma client
cd packages/database && npx prisma generate

# View database
cd packages/database && npx prisma studio
```

---

## 📊 Success Metrics (First 3 Months)

### Primary
- **Meetup Conversion Rate** > 15%
- **7D Retention** > 40%
- **Data Yield/User** > 5 labeled interactions

### Secondary
- **Service Intent Rate** > 8%
- **Event Attendance Rate** > 60%
- **Verified Users** > 30%
- **Avg Streak** > 2 weeks

---

## 🙏 Acknowledgments

Built with high standards, systematic approach, and focus on data collection for ML superiority.

**Tech Stack**: NestJS, Next.js, Prisma, PostgreSQL, TypeScript, Tailwind CSS, shadcn/ui

**Generated with**: [Claude Code](https://claude.com/claude-code)

---

**Last Updated**: October 5, 2025
**Version**: MVP Beta v1.0
**Status**: ✅ Feature Complete - Ready for Testing
