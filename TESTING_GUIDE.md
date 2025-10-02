# Testing Guide - Woof Frontend

## Prerequisites

Make sure both services are running:

```bash
# Terminal 1: API (should already be running)
pnpm --filter @woof/api dev

# Terminal 2: Web Frontend (should already be running)
pnpm --filter @woof/web dev
```

## Quick Health Check

### 1. Check Services Status

```bash
# Check if both ports are listening
lsof -i :3000 -i :4000

# Expected output:
# node ... *:3000 (LISTEN)  <- Next.js
# node ... *:4000 (LISTEN)  <- NestJS
```

### 2. Test API

```bash
# Health check
curl http://localhost:4000/api/health

# Should return: {"status":"ok"}

# Login with demo user
curl -X POST http://localhost:4000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@woof.com","password":"password123"}'

# Should return JWT token and user data
```

## Frontend Testing

### 1. Open in Browser

Visit: **http://localhost:3000**

### 2. Test All Pages

Click through the navigation and verify each page loads:

#### ✅ Dashboard (/)
- [ ] Welcome section with happiness score displays
- [ ] Pet cards show (Buddy, Luna, Max from seed data)
- [ ] Weekly progress shows activity stats
- [ ] All gradients and styling look correct

#### ✅ Pets (/pets)
- [ ] Header shows "My Pets" title
- [ ] Pet grid displays all pets
- [ ] Each pet card shows:
  - Pet name and breed
  - Energy level badge
  - Today's activity stats
  - Progress bar

#### ✅ Activities (/activities)
- [ ] Activity feed displays
- [ ] Each activity shows:
  - Activity type icon
  - Pet name
  - Distance/Duration/Calories
  - Relative time (e.g., "2h ago")

#### ✅ Social (/social)
- [ ] Happiness metrics grid shows 4 metrics
- [ ] Social feed displays posts
- [ ] Each post shows:
  - User avatar and handle
  - Post content
  - Like/comment/share buttons
  - Pet species badge

#### ✅ Leaderboard (/leaderboard)
- [ ] Global leaderboard shows rankings
- [ ] Rank badges show correct colors:
  - 🥇 Rank 1: Gold
  - 🥈 Rank 2: Silver
  - 🥉 Rank 3: Bronze
- [ ] Local leaderboard displays
- [ ] Points display correctly

### 3. Test Responsive Design

#### Desktop (resize browser)
- [ ] Header navigation visible
- [ ] Layout uses 2-3 columns for grids
- [ ] All text readable
- [ ] No horizontal scroll

#### Mobile (resize to ~375px width)
- [ ] Bottom mobile navigation visible
- [ ] Single column layout
- [ ] Header simplified
- [ ] Touch targets adequate size

### 4. Test Interactive Elements

#### Navigation
- [ ] Click each nav link in header
- [ ] Click each nav link in mobile bottom bar
- [ ] Active page highlights correctly
- [ ] All routes work

#### User Menu
- [ ] Click avatar in header
- [ ] Dropdown menu appears
- [ ] Shows user info (Demo User)
- [ ] Shows points (2,450 pts)
- [ ] Profile/Logout buttons visible

#### Social Feed (if posts exist)
- [ ] Click heart icon on a post
- [ ] Like count increments
- [ ] Heart icon turns red
- [ ] Click again to unlike
- [ ] Count decrements

### 5. Test Theme

#### Galaxy-Dark Theme
- [ ] Background is dark (#0E1220)
- [ ] Primary color is deep blue (#0B1C3D)
- [ ] Accent color is bright blue (#6BA8FF)
- [ ] Neuron pattern visible on background
- [ ] Gradients render smoothly
- [ ] Text is readable on dark background

#### Fonts
- [ ] Headings use Space Grotesk
- [ ] Body text uses Inter
- [ ] Font weights look correct

## Browser Testing

Test in multiple browsers:

### Chrome/Edge ✅
```bash
# Open in Chrome
open -a "Google Chrome" http://localhost:3000
```

### Firefox ✅
```bash
# Open in Firefox
open -a Firefox http://localhost:3000
```

### Safari ✅
```bash
# Open in Safari
open -a Safari http://localhost:3000
```

### Mobile Browsers
Use browser DevTools to test:
1. Open DevTools (F12 or Cmd+Option+I)
2. Click device toolbar icon (Cmd+Shift+M)
3. Select iPhone or Android device
4. Test navigation and interactions

## API Integration Testing

### 1. Check Network Requests

1. Open DevTools → Network tab
2. Navigate to Dashboard
3. Verify you see:
   - `GET /api/v1/pets` - Status 200
   - `GET /api/v1/activities` - Status 200

### 2. Test Error Handling

```bash
# Stop the API server
# Navigate to any page
# Should see loading states or error messages (if implemented)
```

## Performance Testing

### 1. Check Load Times

In DevTools → Network tab:
- [ ] First load < 3s
- [ ] Page transitions < 500ms
- [ ] Images load progressively

### 2. Check Console

Open DevTools → Console:
- [ ] No JavaScript errors
- [ ] No warning messages (warnings about dev mode are OK)

## Common Issues & Solutions

### Issue: Page shows "Cannot GET /"
**Solution**: Make sure web dev server is running:
```bash
pnpm --filter @woof/web dev
```

### Issue: "Failed to fetch" errors
**Solution**: Make sure API is running:
```bash
pnpm --filter @woof/api dev
```

### Issue: Data not loading
**Solution**: Check API URL in `.env.local`:
```bash
cat apps/web/.env.local
# Should show: NEXT_PUBLIC_API_URL=http://localhost:4000/api/v1
```

### Issue: Database empty
**Solution**: Re-seed database:
```bash
pnpm --filter @woof/database db:seed
```

### Issue: Port 3000 in use
**Solution**: Kill existing process:
```bash
lsof -ti:3000 | xargs kill -9
pnpm --filter @woof/web dev
```

## Quick Test Script

Run this to test all endpoints:

```bash
#!/bin/bash

echo "Testing Woof Frontend Integration..."

# Test frontend
echo "✓ Testing frontend..."
curl -s http://localhost:3000 > /dev/null && echo "  Frontend: OK" || echo "  Frontend: FAILED"

# Test API
echo "✓ Testing API..."
curl -s http://localhost:4000/api/health > /dev/null && echo "  API: OK" || echo "  API: FAILED"

# Test API endpoints
echo "✓ Testing API endpoints..."
curl -s http://localhost:4000/api/v1/pets > /dev/null && echo "  Pets endpoint: OK" || echo "  Pets endpoint: FAILED"
curl -s http://localhost:4000/api/v1/activities > /dev/null && echo "  Activities endpoint: OK" || echo "  Activities endpoint: FAILED"

echo "✓ All tests complete!"
```

## Visual Testing Checklist

### Colors
- [ ] Dark backgrounds look professional
- [ ] Blue accents pop against dark background
- [ ] Text contrast is readable
- [ ] Gradients are smooth

### Layout
- [ ] Spacing feels balanced
- [ ] Cards have consistent padding
- [ ] Grids align properly
- [ ] No overlapping elements

### Typography
- [ ] Headings are bold and clear
- [ ] Body text is readable
- [ ] Font sizes are appropriate
- [ ] Line heights are comfortable

### Interactions
- [ ] Hover effects work
- [ ] Buttons have clear states
- [ ] Links are obvious
- [ ] Loading states show (if data is slow)

## Success Criteria

✅ All pages load without errors
✅ Navigation works in both desktop and mobile
✅ API data displays correctly
✅ Theme looks professional and consistent
✅ No console errors
✅ Performance is snappy

---

**Ready for production?** Check [DEPLOYMENT.md](./DEPLOYMENT.md) (when available)

**Found a bug?** Create an issue or fix it and commit!
