# PetPath Deployment Guide

## Quick Setup

### 1. Initialize Git Repository
```bash
cd /home/z/my-project
git init
git add .
git commit -m "Initial commit: Complete PetPath platform"
```

### 2. Create GitHub Repository
1. Go to GitHub and create new repository named `petpath`
2. Link and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/petpath.git
git push -u origin main
```

### 3. Set Up GitHub Secrets
Go to repository Settings → Secrets → Actions and add:
- `SECRET_KEY`: `openssl rand -hex 32`
- `DATABASE_URL`: PostgreSQL connection string
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `REDIS_URL`

## Local Testing

### Option A: Docker Compose (Recommended)
```bash
docker-compose -f infra/docker-compose.yml up -d
```
Access: http://localhost:3000 (frontend), http://localhost:8000 (backend)

### Option B: Local Development
```bash
# Frontend
cd frontend && npm install && npm run dev

# Backend  
cd backend && pip install -r requirements.txt && uvicorn main:app --reload

# ML Service
cd ml && pip install -r requirements.txt && uvicorn infer:app --reload
```

## Deployment

### Frontend (Vercel)
1. Install Vercel CLI: `npm install -g vercel`
2. Deploy: `cd frontend && vercel`
3. Configure environment variables in Vercel dashboard

### Backend Services
Choose one:

**Railway (Easiest)**
- Connect GitHub repository to Railway
- Deploy backend and ML services separately

**AWS/Google Cloud**
- Build Docker images
- Deploy to ECS/Cloud Run

### Databases
- **PostgreSQL**: Heroku Postgres, AWS RDS, or Railway
- **Neo4j**: Neo4j Aura (free tier available)
- **Redis**: Redis Labs or AWS ElastiCache

## Testing with Demo Users

### Test Scenarios
1. **User Management**: Registration, login, profiles
2. **Pet Management**: Add, edit, delete pets
3. **Activities**: Log walks, play sessions, track progress
4. **Social Features**: Create posts, like/comment, follow users
5. **Gamification**: Points, leaderboards, achievements
6. **Location**: Nearby users, meetup suggestions

### Demo User Types
- **Power User**: High activity, many friends
- **Casual User**: Moderate activity  
- **New User**: Minimal activity

## Monitoring

Set up:
- Application monitoring (Sentry/New Relic)
- Database performance alerts
- Error tracking
- Regular dependency updates

## Troubleshooting

- **Build fails**: Check Node.js/Python versions, dependencies
- **Database issues**: Verify connection strings, service status
- **API errors**: Check logs, environment variables
- **Frontend issues**: Browser console, API accessibility

## Support

- Create GitHub issues for bugs/features
- Check `/docs` folder for documentation
- Join community channels for support

Your PetPath platform is now ready for testing! 🐾