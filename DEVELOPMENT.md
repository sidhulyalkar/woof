# Woof Development Guide

Complete guide for developing the Woof platform.

## Quick Start

### 1. Prerequisites

- Node.js 20+
- pnpm 8+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)

### 2. Initial Setup

```bash
# Install pnpm globally
npm install -g pnpm

# Install all dependencies
pnpm install

# Start Docker services (PostgreSQL + Redis + n8n)
docker compose up -d

# Wait for PostgreSQL to be ready
docker compose logs postgres

# Set up database environment
cp packages/database/.env.example packages/database/.env
# Edit packages/database/.env:
# DATABASE_URL="postgresql://postgres:postgres@localhost:5432/woof"

# Generate Prisma client
pnpm --filter @woof/database db:generate

# Run migrations
pnpm --filter @woof/database db:migrate

# Seed demo data
pnpm --filter @woof/database db:seed

# Set up API environment
cp apps/api/.env.example apps/api/.env
# Edit apps/api/.env with your configuration
```

### 3. Start Development

```bash
# Start API
pnpm --filter @woof/api dev

# API will be available at:
# http://localhost:4000 - API endpoints
# http://localhost:4000/docs - Swagger documentation
```

---

## Project Structure

```
woof/
├── apps/
│   ├── api/           # NestJS backend ✅
│   ├── web/           # Next.js frontend (planned)
│   └── mobile/        # Expo app (planned)
├── packages/
│   ├── ui/            # Brand system ✅
│   ├── database/      # Prisma schema ✅
│   └── config/        # Shared configs ✅
├── infra/
│   ├── db/            # Database scripts
│   └── docker/        # Docker configs
└── docker-compose.yml # Local dev services
```

---

## Development Workflows

### Working with the Database

```bash
# Generate Prisma client after schema changes
pnpm --filter @woof/database db:generate

# Create a new migration
pnpm --filter @woof/database db:migrate

# Reset database (WARNING: deletes all data)
pnpm --filter @woof/database db:reset

# Open Prisma Studio (database GUI)
pnpm --filter @woof/database db:studio

# Re-seed database
pnpm --filter @woof/database db:seed
```

### Working with the API

```bash
# Development mode (hot reload)
pnpm --filter @woof/api dev

# Build for production
pnpm --filter @woof/api build

# Run production build
pnpm --filter @woof/api start

# Run tests
pnpm --filter @woof/api test

# Run tests in watch mode
pnpm --filter @woof/api test:watch

# Generate test coverage
pnpm --filter @woof/api test:cov

# Lint code
pnpm --filter @woof/api lint
```

### Docker Services

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f postgres
docker compose logs -f redis
docker compose logs -f n8n

# Stop services
docker compose down

# Remove volumes (clean slate)
docker compose down -v

# Restart a service
docker compose restart postgres
```

---

## Testing the API

### Using Swagger UI

1. Start the API: `pnpm --filter @woof/api dev`
2. Open http://localhost:4000/docs
3. Use the interactive Swagger interface

### Using cURL

```bash
# Register a new user
curl -X POST http://localhost:4000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "handle": "testuser",
    "email": "test@example.com",
    "password": "password123",
    "bio": "Test user"
  }'

# Login
curl -X POST http://localhost:4000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'

# Get current user (replace TOKEN with actual JWT)
curl http://localhost:4000/api/v1/auth/me \
  -H "Authorization: Bearer TOKEN"

# Create a pet
curl -X POST http://localhost:4000/api/v1/pets \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Buddy",
    "species": "DOG",
    "breed": "Golden Retriever",
    "sex": "MALE"
  }'

# Get all pets
curl http://localhost:4000/api/v1/pets \
  -H "Authorization: Bearer TOKEN"
```

### Using Demo Data

After seeding, you can login with:

```bash
# Demo user 1
Email: demo@woof.com
Password: password123

# Demo user 2
Email: sarah@woof.com
Password: password123

# Demo user 3
Email: john@woof.com
Password: password123
```

---

## Database Schema

### Key Models

**User**
- `id` - UUID primary key
- `handle` - Unique username
- `email` - Unique email
- `passwordHash` - Bcrypt hash
- `bio` - User bio
- `points` - Gamification points
- Relations: pets, activities, posts

**Pet**
- `id` - UUID primary key
- `ownerId` - User reference
- `name` - Pet name
- `species` - DOG, CAT, etc.
- `breed` - Breed string
- `embedding` - vector(384) for ML
- Relations: owner, activities, edges

**Activity**
- `id` - UUID primary key
- `userId` - User reference
- `petId` - Pet reference
- `type` - WALK, RUN, PLAY, HIKE
- `route` - GeoJSON for tracking
- `humanMetrics` - JSON stats
- `petMetrics` - JSON stats

**PetEdge** (Social Graph)
- `petAId`, `petBId` - Pet references
- `compatibilityScore` - 0-1 score
- `status` - PROPOSED, CONFIRMED, AVOID
- Used for ML recommendations

---

## Common Issues

### Port Already in Use

```bash
# Find process using port 4000
lsof -ti:4000

# Kill process
kill -9 $(lsof -ti:4000)
```

### Database Connection Error

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check connection
psql postgresql://postgres:postgres@localhost:5432/woof

# Restart PostgreSQL
docker compose restart postgres
```

### Prisma Client Not Generated

```bash
# Regenerate Prisma client
pnpm --filter @woof/database db:generate
```

### Module Not Found Errors

```bash
# Clean install
rm -rf node_modules
pnpm install
```

---

## Environment Variables

### API (.env)

```bash
# Server
NODE_ENV=development
PORT=4000
API_PREFIX=api/v1

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/woof

# JWT
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRES_IN=7d

# CORS
CORS_ORIGIN=http://localhost:3000

# Redis
REDIS_URL=redis://localhost:6379
```

### Database (.env)

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/woof
```

---

## Production Deployment

### Build API

```bash
pnpm --filter @woof/api build
```

### Docker Build

```bash
docker build -f infra/docker/Dockerfile.api -t woof-api .
docker run -p 4000:4000 woof-api
```

### Environment Setup

1. Set production DATABASE_URL
2. Set strong JWT_SECRET
3. Configure CORS_ORIGIN
4. Set NODE_ENV=production

---

## Useful Commands

```bash
# Format all code
pnpm format

# Lint all packages
pnpm lint

# Build all packages
pnpm build

# Clean all builds
pnpm clean

# Run all tests
pnpm test
```

---

## Tips

1. **Use Prisma Studio** for database inspection: `pnpm db:studio`
2. **Check Swagger docs** for API endpoints: http://localhost:4000/docs
3. **Monitor Docker logs** when debugging: `docker compose logs -f`
4. **Reset database** if schema gets out of sync: `pnpm db:reset`
5. **Use demo users** for quick testing after seeding

---

## Next Steps

- [ ] Add Socket.io real-time features
- [ ] Implement ML compatibility algorithm
- [ ] Add file upload for avatars
- [ ] Build frontend (apps/web)
- [ ] Create mobile app (apps/mobile)
- [ ] Set up CI/CD pipeline
- [ ] Add comprehensive test coverage

---

**Questions?** Check the README files in each package/app directory.
