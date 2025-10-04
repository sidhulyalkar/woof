# Woof Frontend Implementation Plan

## Phase 2 Status: ✅ COMPLETE

### Infrastructure Built:
- ✅ Zustand stores (session + UI state management)
- ✅ React Query setup with API client
- ✅ Authentication flow with token refresh
- ✅ API hooks for all major features
- ✅ Environment configuration
- ✅ Centered glass morphism layout
- ✅ All Figma screen components

---

## Phase 3-7: Feature Implementation (Multi-Sprint)

### **Sprint 1: Core Social Features (2-3 weeks)**
**Goal**: Users can post, interact, and view their feed

#### 3.1 Feed & Posting System
**Files to Create/Update**:
- `components/feed/CreatePostModal.tsx` - Post creation with image upload
- `components/feed/PostCard.tsx` - Individual post component with like/comment
- `components/feed/StoryCircle.tsx` - Story viewer component
- `components/feed/HighlightCarousel.tsx` - Highlights section

**Backend Integration**:
- Connect `useFeed()` to FeedScreen
- Implement `useCreatePost()` with image upload
- Add `useLikePost()` and comment functionality
- Real-time updates with polling or websockets

**Acceptance Criteria**:
- [ ] Users can create posts with text, images, and location
- [ ] Posts display with user/pet info, likes, comments
- [ ] Like button updates optimistically
- [ ] Stories appear at top of feed
- [ ] Can view and create highlights

---

#### 3.2 Profile Pages (User & Pet)
**Files to Create/Update**:
- `components/profile/UserProfileScreen.tsx` - Replace current ProfileScreen
- `components/profile/PetProfileScreen.tsx` - Dedicated pet profile
- `components/profile/EditProfileModal.tsx` - Edit user info
- `components/profile/EditPetModal.tsx` - Edit pet info
- `components/profile/PostGrid.tsx` - User's posts in grid view
- `components/profile/StatsCard.tsx` - Activity stats display

**Backend Integration**:
- Connect `useUserProfile(userId)` and `usePetProfile(petId)`
- Update profile mutations
- Fetch user's posts and activities

**Acceptance Criteria**:
- [ ] View own profile with stats, posts, pets
- [ ] View other users' profiles
- [ ] Edit profile info and avatar
- [ ] View and edit pet profiles
- [ ] Display activity statistics

---

### **Sprint 2: Activity & Health Tracking (2-3 weeks)**
**Goal**: Track walks/runs with maps, log health data

#### 4.1 Activity Tracking & Maps
**Files to Create/Update**:
- `components/activity/MapScreen.tsx` - Replace current ActivityScreen
- `components/activity/LiveTrackingModal.tsx` - Active walk/run tracker
- `components/activity/RouteMap.tsx` - Display route on map
- `components/activity/ActivityLogger.tsx` - Manual activity entry
- `components/activity/POIMarker.tsx` - Points of interest on map

**New Packages**:
```bash
pnpm add mapbox-gl react-map-gl @turf/turf
```

**Backend Integration**:
- `useActivities()` for history
- `useCreateActivity()` for logging
- Geolocation API for live tracking
- Save routes as GeoJSON

**Acceptance Criteria**:
- [ ] Start live activity tracking (walk/run)
- [ ] Display route on interactive map
- [ ] Show real-time distance, pace, duration
- [ ] Save completed activities
- [ ] View activity history with maps
- [ ] Display POIs (parks, dog parks, vet clinics)

---

#### 4.2 Health Dashboard
**Files to Create/Update**:
- `components/health/HealthDashboard.tsx` - Main health screen
- `components/health/VitalStats.tsx` - Weight, temp, heart rate
- `components/health/MedicationTracker.tsx` - Med schedule
- `components/health/VetVisitLog.tsx` - Appointment history
- `components/health/HealthChart.tsx` - Weight/vitals over time

**New Packages**:
```bash
pnpm add recharts
```

**Acceptance Criteria**:
- [ ] Log pet vitals (weight, temperature)
- [ ] Track medications with reminders
- [ ] Log vet visits and vaccinations
- [ ] View health trends with charts
- [ ] Export health records

---

### **Sprint 3: Social Discovery & Messaging (2 weeks)**
**Goal**: Find friends, chat with other pet owners

#### 5.1 Friends & Discover
**Files to Create/Update**:
- `components/discover/DiscoverScreen.tsx` - Browse users/pets
- `components/discover/SearchBar.tsx` - Search with filters
- `components/discover/UserCard.tsx` - User preview card
- `components/discover/FriendRequestsList.tsx` - Pending requests
- `components/friends/FriendsListScreen.tsx` - All friends view

**Backend Integration**:
- `useFriends()` for friends list
- `useAddFriend()` and friend request system
- Search API with filters (breed, location, activity level)

**Acceptance Criteria**:
- [ ] Search for users by username, location
- [ ] Filter by pet breed, age, activity level
- [ ] Send/accept/decline friend requests
- [ ] View friends list
- [ ] Suggested friends based on location/activity

---

#### 5.2 Messaging System
**Files to Create/Update**:
- `components/messages/ConversationsList.tsx` - Update MessagesScreen
- `components/messages/ChatScreen.tsx` - Individual conversation
- `components/messages/MessageBubble.tsx` - Message component
- `components/messages/MessageInput.tsx` - Send messages with media
- `components/messages/ReactionPicker.tsx` - Emoji reactions

**New Packages**:
```bash
pnpm add socket.io-client emoji-picker-react
```

**Backend Integration**:
- `useConversations()` and `useConversation(id)`
- `useSendMessage()` mutation
- WebSocket for real-time messaging
- Upload images in messages

**Acceptance Criteria**:
- [ ] View all conversations
- [ ] Send text messages in real-time
- [ ] Send images/videos in messages
- [ ] React to messages with emojis
- [ ] Typing indicators
- [ ] Read receipts
- [ ] Message search

---

### **Sprint 4: Leaderboards & Gamification (1-2 weeks)**
**Goal**: Competitive features, achievements, rewards

#### 6.1 Enhanced Leaderboards
**Files to Create/Update**:
- Update `components/LeaderboardScreen.tsx` with backend data
- `components/leaderboard/FilterTabs.tsx` - Weekly/Monthly/All-time
- `components/leaderboard/CategoryFilter.tsx` - Distance/Time/Calories
- `components/leaderboard/LeaderboardCard.tsx` - Rank display

**Backend Integration**:
- Connect `useLeaderboard(timeframe)` to LeaderboardScreen
- Fetch by category (distance, duration, consistency)

**Acceptance Criteria**:
- [ ] View leaderboards by timeframe (weekly/monthly)
- [ ] Filter by category (distance, time, calories)
- [ ] Show user's current rank
- [ ] Friends-only leaderboard option
- [ ] Top 3 podium display

---

#### 6.2 Goals & Achievements
**Files to Create/Update**:
- `components/goals/GoalsScreen.tsx` - Set and track goals
- `components/goals/GoalCard.tsx` - Individual goal display
- `components/goals/CreateGoalModal.tsx` - Goal creation
- `components/achievements/AchievementsList.tsx` - All achievements
- `components/achievements/AchievementUnlockedModal.tsx` - Celebration

**Acceptance Criteria**:
- [ ] Set daily/weekly activity goals
- [ ] Track progress toward goals
- [ ] Unlock achievements (badges)
- [ ] Celebrate achievements with animations
- [ ] View all locked/unlocked achievements

---

### **Sprint 5: Camera & Content Creation (2 weeks)**
**Goal**: Capture photos/videos, create stories and highlights

#### 7.1 Camera & Photo Features
**Files to Create/Update**:
- `components/camera/CameraModal.tsx` - Full camera interface
- `components/camera/CameraControls.tsx` - Photo/video/story modes
- `components/camera/FilterSelector.tsx` - Photo filters
- `components/camera/PhotoEditor.tsx` - Basic editing (crop, adjust)

**New Packages**:
```bash
pnpm add react-webcam @ffmpeg/ffmpeg
```

**Backend Integration**:
- `useUploadImage()` for photos
- Video upload support
- Story creation endpoint

**Acceptance Criteria**:
- [ ] Take photos with device camera
- [ ] Record videos (up to 60s)
- [ ] Apply filters before posting
- [ ] Basic editing (crop, brightness, contrast)
- [ ] Upload to posts or stories

---

#### 7.2 Stories & Highlights
**Files to Create/Update**:
- `components/story/StoryViewer.tsx` - Full-screen story viewer
- `components/story/StoryProgress.tsx` - Progress bar
- `components/story/CreateStoryModal.tsx` - Story creation flow
- `components/highlights/HighlightCreator.tsx` - Convert stories to highlights

**Acceptance Criteria**:
- [ ] Post 24-hour stories
- [ ] View friends' stories in sequence
- [ ] Story progress indicators
- [ ] Save stories to highlights
- [ ] Organize highlights by category

---

### **Sprint 6: Advanced Features (2-3 weeks)**
**Goal**: Video editing, advanced maps, mental health

#### 8.1 Video Editing
**Files to Create/Update**:
- `components/video/VideoEditor.tsx` - Trim, merge clips
- `components/video/VideoTimeline.tsx` - Timeline scrubber
- `components/video/VideoEffects.tsx` - Transitions, text overlays

**Acceptance Criteria**:
- [ ] Trim video clips
- [ ] Merge multiple clips
- [ ] Add text overlays
- [ ] Apply transitions
- [ ] Export edited videos

---

#### 8.2 Advanced Map Features
**Files to Create/Update**:
- `components/map/MeetupCreator.tsx` - Create meetup events
- `components/map/MeetupCard.tsx` - Meetup details
- `components/map/RouteBuilder.tsx` - Plan walking routes
- `components/map/RouteLibrary.tsx` - Save/share favorite routes

**Acceptance Criteria**:
- [ ] Create meetup events at locations
- [ ] RSVP to meetups
- [ ] View nearby meetups on map
- [ ] Plan and save walking routes
- [ ] Share routes with friends
- [ ] Rate routes and leave reviews

---

#### 8.3 Mental Health Dashboard
**Files to Create/Update**:
- `components/mental-health/MoodTracker.tsx` - Daily mood logging
- `components/mental-health/MoodChart.tsx` - Mood trends
- `components/mental-health/StressMonitor.tsx` - Stress indicators
- `components/mental-health/MindfulnessTimer.tsx` - Meditation timer

**Acceptance Criteria**:
- [ ] Log daily mood
- [ ] Track stress levels
- [ ] View mood trends over time
- [ ] Guided breathing exercises
- [ ] Mindfulness timer
- [ ] Correlate mood with activity levels

---

### **Sprint 7: Settings & Polish (1-2 weeks)**
**Goal**: Privacy, safety, notifications, final polish

#### 9.1 Settings & Privacy
**Files to Create/Update**:
- `components/settings/SettingsScreen.tsx` - Main settings
- `components/settings/PrivacySettings.tsx` - Privacy controls
- `components/settings/NotificationSettings.tsx` - Notification prefs
- `components/settings/BlockedUsersList.tsx` - Blocked accounts
- `components/settings/DataExport.tsx` - Export user data

**Acceptance Criteria**:
- [ ] Privacy controls (profile visibility, location)
- [ ] Block/report users
- [ ] Notification preferences
- [ ] Account settings (email, password)
- [ ] Data export (GDPR compliance)
- [ ] Delete account option

---

#### 9.2 Safety Features
**Files to Create/Update**:
- `components/safety/SafetyCheckIn.tsx` - Periodic check-ins during activities
- `components/safety/EmergencyContact.tsx` - Emergency contact management
- `components/safety/LocationSharing.tsx` - Share live location

**Acceptance Criteria**:
- [ ] Set up emergency contacts
- [ ] Share live location with trusted contacts
- [ ] Safety check-ins during long activities
- [ ] Report inappropriate content
- [ ] Safety tips and guidelines

---

## Technical Debt & Optimization

### Performance Optimization
- [ ] Implement virtual scrolling for long feeds
- [ ] Image lazy loading and optimization
- [ ] Code splitting for routes
- [ ] Service worker for offline support
- [ ] React Query cache optimization

### Testing
- [ ] Unit tests for utility functions
- [ ] Integration tests for API hooks
- [ ] E2E tests for critical flows (auth, posting, messaging)
- [ ] Visual regression testing

### Documentation
- [ ] Component documentation
- [ ] API integration guide
- [ ] Deployment guide
- [ ] User guide

---

## Deployment Checklist

### Pre-Production
- [ ] Environment variables configured
- [ ] Error tracking (Sentry)
- [ ] Analytics integration
- [ ] Performance monitoring
- [ ] SSL certificates
- [ ] CDN setup for media

### Production Launch
- [ ] Database migrations
- [ ] Backend API deployed
- [ ] Frontend deployed
- [ ] Domain configured
- [ ] Monitoring dashboards
- [ ] Backup strategy

---

## Estimated Timeline
- **Sprint 1**: Core Social - 3 weeks
- **Sprint 2**: Activity & Health - 3 weeks
- **Sprint 3**: Social Discovery & Messaging - 2 weeks
- **Sprint 4**: Leaderboards & Gamification - 2 weeks
- **Sprint 5**: Camera & Content - 2 weeks
- **Sprint 6**: Advanced Features - 3 weeks
- **Sprint 7**: Settings & Polish - 2 weeks

**Total: ~17 weeks (4+ months) for MVP**

---

## Next Immediate Steps

1. **Sprint 1 Kickoff**: Start with Feed & Posting System
2. Install required packages for Sprint 1
3. Create feed components with backend integration
4. Build profile pages
5. Test and iterate

Ready to begin Sprint 1?
