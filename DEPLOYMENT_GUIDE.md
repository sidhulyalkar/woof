# Woof App Deployment Guide

Complete guide for deploying the Woof app to production for user testing.

---

## 📋 Prerequisites

- [ ] Vercel account
- [ ] Fly.io or Railway account
- [ ] Neon or Supabase account (PostgreSQL)
- [ ] GitHub repository with latest code
- [ ] Environment variables prepared

---

## 🗄️ Step 1: Deploy Database

### Option A: Neon (Recommended)

1. Go to [neon.tech](https://neon.tech)
2. Create new project: "woof-production"
3. Copy connection string
4. Run migrations:

```bash
# Set DATABASE_URL in packages/database/.env
DATABASE_URL="postgresql://user:pass@host/woof?sslmode=require"

# Run migrations
pnpm --filter @woof/database db:migrate

# Seed production data
pnpm --filter @woof/database db:seed
```

### Option B: Supabase

1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Enable pgvector extension in SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
4. Copy connection string from Settings > Database
5. Run migrations as above

---

## 🚀 Step 2: Deploy Backend API

### Option A: Fly.io (Recommended)

1. Install Fly CLI:
```bash
brew install flyctl
# or
curl -L https://fly.io/install.sh | sh
```

2. Login and create app:
```bash
flyctl auth login
cd apps/api
flyctl launch --name woof-api --region sjc
```

3. Set environment variables:
```bash
flyctl secrets set \
  NODE_ENV=production \
  PORT=8080 \
  API_PREFIX=api/v1 \
  DATABASE_URL="postgresql://..." \
  JWT_SECRET="$(openssl rand -base64 32)" \
  JWT_REFRESH_SECRET="$(openssl rand -base64 32)" \
  CORS_ORIGIN="https://woof.vercel.app"
```

4. Create `fly.toml` in apps/api:
```toml
app = "woof-api"
primary_region = "sjc"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

5. Deploy:
```bash
flyctl deploy
```

6. Verify:
```bash
curl https://woof-api.fly.dev/api/v1/health
```

### Option B: Railway

1. Go to [railway.app](https://railway.app)
2. Create new project from GitHub repo
3. Select `apps/api` as root directory
4. Add environment variables in Settings
5. Deploy automatically on git push

---

## 🌐 Step 3: Deploy Frontend

### Vercel (Recommended)

1. Go to [vercel.com](https://vercel.com)
2. Import from GitHub
3. Configure project:
   - **Framework Preset**: Next.js
   - **Root Directory**: apps/web
   - **Build Command**: `cd ../.. && pnpm install && pnpm --filter @woof/web build`
   - **Output Directory**: apps/web/.next

4. Add environment variables:
```
NEXT_PUBLIC_API_URL=https://woof-api.fly.dev/api/v1
```

5. Deploy and verify

---

## ✅ Step 4: Post-Deployment Verification

### 1. API Health Check
```bash
curl https://woof-api.fly.dev/api/v1/health
# Should return 200 OK
```

### 2. Swagger Documentation
Visit: `https://woof-api.fly.dev/docs`

### 3. Frontend Access
Visit: `https://woof.vercel.app`

### 4. Test Authentication
1. Open frontend
2. Click "Create Account"
3. Register new user
4. Verify login works
5. Check API logs for authentication

### 5. Test API Integration
1. View feed (should fetch from API)
2. Create post (if implemented)
3. View events
4. Check browser DevTools for API calls

---

## 🔧 Step 5: Configure Domain (Optional)

### Backend Custom Domain
```bash
flyctl certs add api.woofapp.com
```

Update CORS in backend:
```env
CORS_ORIGIN=https://app.woofapp.com,https://woofapp.com
```

### Frontend Custom Domain
1. Go to Vercel project settings
2. Add custom domain: `woofapp.com` or `app.woofapp.com`
3. Configure DNS records as instructed

---

## 📊 Step 6: Monitoring & Analytics

### 1. Sentry (Error Tracking)

**Backend:**
```bash
pnpm add @sentry/node
```

```typescript
// apps/api/src/main.ts
import * as Sentry from '@sentry/node';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
});
```

**Frontend:**
```bash
pnpm add @sentry/nextjs
```

### 2. Vercel Analytics
Already configured via `@vercel/analytics` package.

### 3. Database Monitoring
- Neon: Built-in dashboard
- Supabase: Built-in dashboard + logs

---

## 🧪 Step 7: Create Test Accounts

Run this script to create beta test users:

```sql
-- Connect to production database
INSERT INTO "User" (id, handle, email, "passwordHash", bio, points)
VALUES
  (gen_random_uuid(), 'tester1', 'tester1@woofapp.com', '$2b$10$...', 'Beta Tester 1', 0),
  (gen_random_uuid(), 'tester2', 'tester2@woofapp.com', '$2b$10$...', 'Beta Tester 2', 0),
  (gen_random_uuid(), 'tester3', 'tester3@woofapp.com', '$2b$10$...', 'Beta Tester 3', 0);
```

Or use the registration endpoint.

---

## 📱 Step 8: PWA Configuration

### 1. Update manifest.json
```json
{
  "name": "Woof - Pet Social Fitness",
  "short_name": "Woof",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0a0e27",
  "theme_color": "#00d9ff",
  "icons": [
    {
      "src": "/icon-192.jpg",
      "sizes": "192x192",
      "type": "image/jpeg"
    },
    {
      "src": "/icon-512.jpg",
      "sizes": "512x512",
      "type": "image/jpeg"
    }
  ]
}
```

### 2. Test PWA
1. Open app on mobile device
2. Look for "Add to Home Screen" prompt
3. Install and test offline functionality

---

## 🚨 Troubleshooting

### API Returns 500 Error
1. Check Fly.io logs: `flyctl logs`
2. Verify DATABASE_URL is correct
3. Check migrations ran successfully

### CORS Error
1. Verify CORS_ORIGIN includes frontend URL
2. Check for trailing slashes
3. Restart API after environment changes

### Frontend Build Fails
1. Check pnpm workspace is set up correctly
2. Verify all dependencies are installed
3. Check build logs in Vercel dashboard

### Database Connection Fails
1. Verify DATABASE_URL format
2. Check SSL mode requirement
3. Test connection locally first

---

## 📈 Scaling Considerations

### Database
- **Connection Pooling**: Use Prisma connection pooling
- **Indexes**: Add indexes for frequently queried fields
- **Backups**: Enable automated backups on Neon/Supabase

### API
- **Horizontal Scaling**: Increase Fly.io machine count
- **Caching**: Enable Redis caching
- **Rate Limiting**: Already configured in backend

### Frontend
- **Image Optimization**: Use Next.js Image component
- **Code Splitting**: Automatic with Next.js
- **Edge Functions**: Use Vercel Edge for dynamic content

---

## 🔐 Security Checklist

- [ ] Change all default secrets
- [ ] Use strong JWT_SECRET (32+ characters)
- [ ] Enable HTTPS only
- [ ] Configure rate limiting
- [ ] Set up CORS properly
- [ ] Enable database SSL
- [ ] Use environment variables (never commit secrets)
- [ ] Set up monitoring alerts

---

## 🎯 Pre-Launch Checklist

- [ ] Database deployed and migrated
- [ ] Backend API deployed and healthy
- [ ] Frontend deployed and accessible
- [ ] Authentication flow working
- [ ] Core features functional
- [ ] PWA installable
- [ ] Error tracking configured
- [ ] Analytics set up
- [ ] Test accounts created
- [ ] Documentation updated

---

## 📞 Support Resources

- **Fly.io**: https://fly.io/docs
- **Vercel**: https://vercel.com/docs
- **Neon**: https://neon.tech/docs
- **Supabase**: https://supabase.com/docs
- **Prisma**: https://www.prisma.io/docs
- **NestJS**: https://docs.nestjs.com

---

## 🎉 Launch Day

1. **Final Smoke Test**
   - Test all critical user flows
   - Verify all API endpoints respond
   - Check mobile responsiveness

2. **Send Invites**
   - Share app URL with beta testers
   - Provide test account credentials
   - Share feedback form

3. **Monitor**
   - Watch error logs in Sentry
   - Monitor API response times
   - Track user signups
   - Respond to feedback

---

**Last Updated**: October 9, 2025
**Status**: Ready for deployment ✅
