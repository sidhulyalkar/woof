# 🚀 Quick Start: Regenerate Prisma Client

## Current Status

✅ **Code Complete** - All nudge engine code is written and ready
⏳ **Database Migration Pending** - Need to update schema and regenerate Prisma client

## Why This Is Needed

The Prisma client is currently out of sync with the updated schema. We added:
- Chat/Messaging models (Conversation, Message, ConversationParticipant)
- Enhanced ProactiveNudge model (dismissed, createdAt fields)

TypeScript errors you're seeing will disappear once Prisma is regenerated.

---

## 🎯 Two Ways to Complete Setup

### Option A: Automated Setup (Recommended)

```bash
# 1. Ensure Docker Desktop is running
open -a Docker  # or start manually

# 2. Run the setup script
chmod +x setup-nudges.sh
./setup-nudges.sh
```

This script will:
- ✅ Check Docker is running
- ✅ Start PostgreSQL and n8n services
- ✅ Wait for database to be ready
- ✅ Run database migrations
- ✅ Generate Prisma client
- ✅ Generate VAPID keys (if not exists)
- ✅ Rebuild API

---

### Option B: Manual Steps

#### Step 1: Start Docker Services

```bash
# Check if Docker is running
docker info

# If not, start Docker Desktop, then:
docker-compose up -d postgres n8n
```

#### Step 2: Wait for PostgreSQL

```bash
# Wait until this command succeeds:
docker-compose exec postgres pg_isready -U woof

# Should output:
# /var/run/postgresql:5432 - accepting connections
```

#### Step 3: Run Migrations

```bash
cd packages/database

# Run migration
npm run db:migrate -- --name add_chat_and_nudge_improvements

# Expected output:
# ✓ Prisma schema loaded
# ✓ Datasource "db": PostgreSQL database "woof"
# ✓ Migration applied successfully
```

#### Step 4: Generate Prisma Client

```bash
# Still in packages/database
npm run db:generate

# Expected output:
# ✔ Generated Prisma Client
```

#### Step 5: Generate VAPID Keys (if needed)

```bash
cd ../..

# Generate keys
npx web-push generate-vapid-keys

# Copy output and add to apps/api/.env:
# VAPID_PUBLIC_KEY=<your-public-key>
# VAPID_PRIVATE_KEY=<your-private-key>

# Also add to apps/web/.env.local:
# NEXT_PUBLIC_VAPID_PUBLIC_KEY=<your-public-key>
```

#### Step 6: Rebuild API

```bash
cd apps/api
npm run build

# TypeScript errors should be gone now!
```

---

## 🧪 Verify It Worked

### Check 1: No TypeScript Errors

```bash
cd apps/api
npm run build

# Should complete without errors
```

### Check 2: Prisma Client Has New Models

```bash
# In Node/TypeScript console or a test file:
import { PrismaClient } from '@prisma/client'
const prisma = new PrismaClient()

// These should work without errors:
prisma.conversation  // ✅ Should exist
prisma.message       // ✅ Should exist
prisma.proactiveNudge // ✅ Should have dismissed field
```

### Check 3: Start Development Servers

```bash
# From root directory
npm run dev

# Both API and Web should start without errors
```

---

## 🐛 Troubleshooting

### Error: "Can't reach database server at localhost:5432"

**Solution**: PostgreSQL isn't running
```bash
docker-compose up -d postgres
# Wait 10 seconds, then retry migration
```

### Error: "Environment variable not found: DATABASE_URL"

**Solution**: .env file not found
```bash
# Check if .env exists:
ls apps/api/.env

# If not, copy from example:
cp apps/api/.env.example apps/api/.env
# Edit with your database credentials
```

### Error: "Docker daemon not running"

**Solution**: Start Docker Desktop
```bash
# On macOS:
open -a Docker

# On Windows:
# Start Docker Desktop from Start Menu

# Wait for Docker to start, then retry
```

### TypeScript errors persist after migration

**Solution**: Clear build cache and rebuild
```bash
# Delete generated files
rm -rf apps/api/dist
rm -rf node_modules/.prisma

# Regenerate
cd packages/database
npm run db:generate

# Rebuild
cd ../../apps/api
npm run build
```

---

## 📋 Environment Variables Checklist

### apps/api/.env
```bash
DATABASE_URL="postgresql://woof:password@localhost:5432/woof"
SHADOW_DATABASE_URL="postgresql://woof:password@localhost:5432/woof_shadow"
VAPID_PUBLIC_KEY="<your-key>"
VAPID_PRIVATE_KEY="<your-key>"
JWT_SECRET="<your-secret>"
```

### apps/web/.env.local
```bash
NEXT_PUBLIC_VAPID_PUBLIC_KEY="<your-key>"
NEXT_PUBLIC_API_URL="http://localhost:3001"
```

---

## ✅ Success Indicators

You'll know it worked when:

1. ✅ `npm run build` in `apps/api` completes without TypeScript errors
2. ✅ Database shows new tables: `conversations`, `messages`, `conversation_participants`
3. ✅ ProactiveNudge table has `dismissed` and `created_at` columns
4. ✅ Development servers start successfully
5. ✅ Push notification test works

---

## 🎯 After Successful Setup

Once Prisma is regenerated, test the nudge system:

```bash
# Terminal 1: Start dev servers
npm run dev

# Terminal 2: Test proximity nudges
curl -X POST http://localhost:3001/api/nudges/check/proximity \
  -H "Authorization: Bearer <your-jwt>"

# Terminal 3: Monitor logs
docker-compose logs -f api
```

---

## 📚 Next Steps After Migration

1. ✅ Test proximity nudges
2. ✅ Test chat activity nudges
3. ✅ Set up push notifications in browser
4. ✅ Create n8n workflows for automation
5. ✅ Add analytics telemetry
6. ✅ Final testing before beta

---

## 🆘 Need Help?

- **Setup Guide**: See [NUDGE_ENGINE_SETUP.md](NUDGE_ENGINE_SETUP.md)
- **Implementation Details**: See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
- **Schema Changes**: Check [packages/database/prisma/schema.prisma](packages/database/prisma/schema.prisma)

**Estimated Time**: 5-10 minutes for automated setup, 15-20 minutes for manual setup.

Good luck! 🐾
