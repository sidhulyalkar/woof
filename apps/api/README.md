# Woof API

NestJS-powered backend for the Woof Pet Social Fitness Platform.

## Features

- ✅ JWT Authentication with refresh tokens
- ✅ User & Pet management
- ✅ Activity tracking (walks, runs, plays)
- ✅ Social features (posts, comments, likes)
- ✅ Meetup coordination
- ✅ ML-powered pet compatibility scoring
- ✅ Automatic Swagger/OpenAPI documentation
- ✅ Prisma ORM with type safety
- ✅ WebSocket support (Socket.io)

## Tech Stack

- **Framework**: NestJS 10
- **Database**: Prisma + PostgreSQL (with pgvector)
- **Auth**: Passport.js + JWT
- **Validation**: class-validator + Zod
- **Documentation**: Swagger/OpenAPI
- **Real-time**: Socket.io

## Quick Start

### 1. Install Dependencies

```bash
pnpm install
```

### 2. Environment Setup

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Database Setup

```bash
# Generate Prisma client
pnpm --filter @woof/database db:generate

# Run migrations
pnpm --filter @woof/database db:migrate

# Seed database
pnpm --filter @woof/database db:seed
```

### 4. Start Development Server

```bash
pnpm dev
```

The API will be available at:
- **API**: http://localhost:4000
- **Swagger Docs**: http://localhost:4000/docs

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login
- `GET /api/v1/auth/me` - Get current user profile

### Users
- `GET /api/v1/users` - Get all users (paginated)
- `GET /api/v1/users/:id` - Get user by ID

### Pets
- `POST /api/v1/pets` - Create pet
- `GET /api/v1/pets` - Get all pets
- `GET /api/v1/pets/:id` - Get pet by ID
- `PUT /api/v1/pets/:id` - Update pet
- `DELETE /api/v1/pets/:id` - Delete pet

### Activities
- `POST /api/v1/activities` - Log activity
- `GET /api/v1/activities` - Get activities
- `GET /api/v1/activities/:id` - Get activity by ID

### Social
- `POST /api/v1/social/posts` - Create post
- `GET /api/v1/social/posts` - Get feed
- `POST /api/v1/social/posts/:postId/likes` - Like post
- `POST /api/v1/social/posts/:postId/comments` - Comment on post

### Meetups
- `POST /api/v1/meetups` - Create meetup
- `GET /api/v1/meetups` - Get meetups
- `POST /api/v1/meetups/invites` - Send invite

### Compatibility
- `POST /api/v1/compatibility/calculate` - Calculate pet compatibility
- `GET /api/v1/compatibility/recommendations/:petId` - Get recommendations

## Development

### Run Tests

```bash
pnpm test
pnpm test:watch
pnpm test:cov
```

### Lint Code

```bash
pnpm lint
```

### Build for Production

```bash
pnpm build
pnpm start
```

## Architecture

```
src/
├── auth/              # Authentication (JWT, Passport)
├── users/             # User management
├── pets/              # Pet profiles
├── activities/        # Activity tracking
├── social/            # Social feed (posts, likes, comments)
├── meetups/           # Meetup coordination
├── compatibility/     # ML compatibility scoring
├── prisma/            # Prisma service
├── app.module.ts      # Root module
└── main.ts            # Application entry point
```

## Database Models

See `packages/database/prisma/schema.prisma` for complete schema.

Key models:
- User
- Pet
- Activity
- Post, Like, Comment
- Meetup, MeetupInvite
- PetEdge (compatibility graph)
- Place, Telemetry

## Authentication

All endpoints (except `/auth/register` and `/auth/login`) require JWT token:

```bash
# Login
curl -X POST http://localhost:4000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@woof.com","password":"password123"}'

# Use token
curl http://localhost:4000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment | `development` |
| `PORT` | Server port | `4000` |
| `DATABASE_URL` | PostgreSQL connection | Required |
| `JWT_SECRET` | JWT signing key | Required |
| `JWT_EXPIRES_IN` | Token expiry | `7d` |
| `CORS_ORIGIN` | Allowed origins | `http://localhost:3000` |

## Deployment

### Docker

```bash
docker build -t woof-api .
docker run -p 4000:4000 woof-api
```

### Fly.io

```bash
fly deploy
```

---

Built with ❤️ using NestJS
