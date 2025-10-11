# Proactive Nudge Engine Setup Guide

## Overview
This guide documents the implementation of the Proactive Meetup Nudge Engine for PetPath/Woof. The system automatically suggests meetups based on proximity, chat activity, and mutual availability.

## ✅ Completed Changes

### 1. Database Schema Updates

#### Added Chat/Messaging Models
```prisma
model Conversation {
  id          String   @id @default(uuid())
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
  participants ConversationParticipant[]
  messages     Message[]
}

model ConversationParticipant {
  id             String   @id @default(uuid())
  conversationId String
  userId         String
  joinedAt       DateTime @default(now())
  lastReadAt     DateTime?
}

model Message {
  id             String   @id @default(uuid())
  conversationId String
  senderId       String
  text           String
  mediaUrls      String[] @default([])
  createdAt      DateTime @default(now())
}
```

#### Updated ProactiveNudge Model
```prisma
model ProactiveNudge {
  id           String    @id @default(uuid())
  userId       String
  targetUserId String?
  type         String    // meetup, event, service, safety_reminder, feedback_request
  payload      Json
  sentVia      String    @default("push") // push, in_app, email
  accepted     Boolean?
  dismissed    Boolean   @default(false)  // NEW
  respondedAt  DateTime?
  createdAt    DateTime  @default(now())  // NEW
  sentAt       DateTime  @default(now())
}
```

### 2. Backend Services

#### NudgesService (`apps/api/src/nudges/nudges.service.ts`)
✅ **Proximity-based nudges** - Cron job runs every 5 minutes to check for nearby compatible users
✅ **Chat activity nudges** - Triggers after 5+ messages in a conversation
✅ **Cooldown enforcement** - 24-hour cooldown between similar nudges
✅ **Push notification integration** - Sends push notifications via web-push

#### NotificationsService (`apps/api/src/notifications/notifications.service.ts`)
✅ **Web Push configured** - Using VAPID keys for PWA push notifications
✅ **Subscription management** - Store/retrieve push subscriptions
✅ **Nudge notifications** - Specialized method for sending nudge alerts
✅ **Achievement notifications** - For gamification events
✅ **Event reminders** - For community events

### 3. API Endpoints

#### Nudge Endpoints
- `GET /nudges` - Get active nudges for current user
- `POST /nudges` - Create manual nudge (admin/testing)
- `PATCH /nudges/:id/dismiss` - Dismiss a nudge
- `PATCH /nudges/:id/accept` - Accept a nudge and take action
- `POST /nudges/check/chat/:conversationId` - Manually trigger chat activity check

#### Notifications Endpoints (implied from service)
- Push subscription management
- Send push notifications
- Bulk notifications

## 🚧 Required Setup Steps

### Step 1: Start Database and Run Migrations

```bash
# Start Docker services (if using Docker)
docker-compose up -d postgres

# Or start your PostgreSQL database however you normally do

# Run migrations from the database package
cd packages/database
npm run db:migrate -- --name add_chat_and_nudge_improvements

# Generate Prisma client
npm run db:generate
```

### Step 2: Set Environment Variables

Ensure these are set in `apps/api/.env`:

```bash
# Database
DATABASE_URL="postgresql://user:password@localhost:5432/woof"
SHADOW_DATABASE_URL="postgresql://user:password@localhost:5432/woof_shadow"

# Web Push (VAPID Keys)
# Generate with: npx web-push generate-vapid-keys
VAPID_PUBLIC_KEY="your-public-key"
VAPID_PRIVATE_KEY="your-private-key"
```

### Step 3: Integrate Chat Service with Nudge Triggers

The chat gateway needs to trigger nudge checks. Add to `apps/api/src/chat/chat.gateway.ts`:

```typescript
import { NudgesService } from '../nudges/nudges.service';

export class ChatGateway {
  constructor(
    private jwtService: JwtService,
    private nudgesService: NudgesService,  // Add this
  ) {}

  @SubscribeMessage('message:send')
  async handleMessage(@ConnectedSocket() client: Socket, @MessageBody() data: ChatMessage) {
    // ... existing code ...

    // Trigger nudge check after message is sent
    await this.nudgesService.checkChatActivityNudges(data.conversationId);

    return { success: true, message };
  }
}
```

### Step 4: Frontend Push Notification Setup

#### Update Service Worker (`apps/web/public/sw.js`)

```javascript
// Add push notification handling
self.addEventListener('push', (event) => {
  const data = event.data.json();

  const options = {
    body: data.body,
    icon: data.icon || '/icon-192.png',
    badge: '/badge-72.png',
    data: data.data,
    actions: data.data?.type === 'nudge' ? [
      { action: 'accept', title: 'Meet Up!' },
      { action: 'dismiss', title: 'Not Now' }
    ] : []
  };

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'accept') {
    // Open app to nudge details
    event.waitUntil(
      clients.openWindow(event.notification.data.url || '/notifications')
    );
  }
});
```

#### Subscribe to Push Notifications

Create `apps/web/src/lib/push-notifications.ts`:

```typescript
export async function subscribeToPushNotifications() {
  if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
    console.warn('Push notifications not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;

    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY
    });

    // Send subscription to backend
    await fetch('/api/notifications/subscribe', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(subscription)
    });

    return subscription;
  } catch (error) {
    console.error('Failed to subscribe to push:', error);
    return null;
  }
}
```

## 📊 How It Works

### Proximity Nudges
1. **Cron job** runs every 5 minutes
2. Checks for co-activity segments in last 10 minutes (users within 50m)
3. Verifies compatibility score ≥ 0.7
4. Checks 24-hour cooldown
5. Creates nudge for both users with distance and pet names
6. Sends push notification

### Chat Activity Nudges
1. Triggered when a message is sent
2. Counts messages in conversation
3. If ≥ 5 messages, suggests meetup
4. Checks 24-hour cooldown
5. Creates nudge with context (message count, conversation ID)
6. Sends push notification

### Nudge Lifecycle
1. **Created** - Nudge is stored in DB with `dismissed: false`
2. **Sent** - Push notification sent to user
3. **Displayed** - User sees nudge in app/notification
4. **Action** - User accepts (redirects to meetup flow) or dismisses
5. **Cooldown** - 24h cooldown prevents spam

## 🔍 Testing

### Test Proximity Nudges
```bash
# Manually trigger proximity check
curl -X POST http://localhost:3001/api/nudges/check/proximity \
  -H "Authorization: Bearer YOUR_JWT"
```

### Test Chat Activity Nudges
```bash
# Trigger chat activity check for a conversation
curl -X POST http://localhost:3001/api/nudges/check/chat/CONVERSATION_ID \
  -H "Authorization: Bearer YOUR_JWT"
```

### Test Push Notifications
```typescript
// In browser console after subscribing
await fetch('/api/notifications/send', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    userId: 'USER_ID',
    title: 'Test Nudge',
    body: 'This is a test meetup suggestion!',
    data: { type: 'nudge', nudgeType: 'meetup' }
  })
});
```

## 🐛 Known Issues & Next Steps

### Current TypeScript Errors
The following errors exist because Prisma client hasn't been regenerated:
- `Property 'conversation' does not exist on type 'PrismaService'`
- `'dismissed' does not exist in type 'ProactiveNudgeWhereInput'`
- `'createdAt' does not exist in type 'ProactiveNudgeWhereInput'`

**Resolution**: Run `npm run db:generate` in `packages/database` after migrations complete.

### Remaining Implementation Tasks
1. ✅ Schema updates complete
2. ✅ Nudge service logic complete
3. ✅ Push notification service complete
4. ⏳ Run database migrations
5. ⏳ Integrate chat gateway with nudge service
6. ⏳ Frontend push subscription UI
7. ⏳ Nudge notification UI component
8. ⏳ Test end-to-end flow

## 📝 API Response Examples

### Get Active Nudges
```json
GET /nudges
Response: [
  {
    "id": "uuid",
    "userId": "user-id",
    "type": "meetup",
    "payload": {
      "targetUserId": "other-user-id",
      "reason": "proximity",
      "message": "BuddyOwner is nearby! Want to meet up?",
      "metadata": {
        "distance": 45,
        "venueType": "park",
        "petNames": {
          "yours": "Buddy",
          "theirs": "Max"
        }
      }
    },
    "dismissed": false,
    "sentAt": "2025-10-10T12:34:56Z"
  }
]
```

### Accept Nudge
```json
PATCH /nudges/:id/accept
Response: {
  "id": "uuid",
  "dismissed": true,
  "payload": { /* nudge context */ }
}
```

## 🔐 Security Considerations

1. **VAPID Keys** - Never commit these to version control
2. **Cooldowns** - Prevent notification spam (24h default)
3. **User Consent** - Request push permission explicitly
4. **Data Privacy** - Location data used only for proximity matching

## 📚 Related Files

- Schema: `packages/database/prisma/schema.prisma`
- Nudges Service: `apps/api/src/nudges/nudges.service.ts`
- Notifications Service: `apps/api/src/notifications/notifications.service.ts`
- Chat Gateway: `apps/api/src/chat/chat.gateway.ts`
- Nudges Controller: `apps/api/src/nudges/nudges.controller.ts`
