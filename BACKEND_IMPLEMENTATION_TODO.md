# Backend Implementation TODO

## ✅ Completed

1. **Prisma Schema** - Added quiz and ML tables
   - `QuizResponse` - Stores quiz submissions
   - `MLFeatureVector` - Stores user feature vectors for matching
   - `UserInteraction` - Tracks likes, matches, meetups, ratings
   - `MLTrainingData` - Stores training data points
   - Migration created and applied successfully

## 🚧 In Progress - Next Steps

### 1. Create Quiz Module (`apps/api/src/quiz/`)

Create the following files:

**quiz.module.ts**
```typescript
import { Module } from '@nestjs/common';
import { QuizService } from './quiz.service';
import { QuizController } from './quiz.controller';

@Module({
  providers: [QuizService],
  controllers: [QuizController],
  exports: [QuizService],
})
export class QuizModule {}
```

**quiz.controller.ts**
```typescript
import { Controller, Post, Body, UseGuards } from '@nestjs/common';
import { QuizService } from './quiz.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { User } from '../decorators/user.decorator';

@Controller('quiz')
@UseGuards(JwtAuthGuard)
export class QuizController {
  constructor(private quizService: QuizService) {}

  @Post('submit')
  async submitQuiz(
    @User() user: any,
    @Body() data: { session: any; featureVector: any },
  ) {
    return this.quizService.submitQuiz(user.id, data.session, data.featureVector);
  }
}
```

**quiz.service.ts**
```typescript
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class QuizService {
  constructor(private prisma: PrismaService) {}

  async submitQuiz(userId: string, session: any, featureVector: any) {
    // 1. Store quiz response
    const quizResponse = await this.prisma.quizResponse.create({
      data: {
        userId,
        petId: session.petId,
        sessionId: session.id,
        responses: session.responses,
        completedAt: session.completedAt,
      },
    });

    // 2. Upsert ML feature vector
    const mlVector = await this.prisma.mLFeatureVector.upsert({
      where: { userId },
      create: {
        userId,
        petId: featureVector.petId,
        features: featureVector.features,
      },
      update: {
        features: featureVector.features,
        petId: featureVector.petId,
      },
    });

    return {
      success: true,
      quizResponseId: quizResponse.id,
      featureVectorId: mlVector.id,
    };
  }

  async getUserFeatureVector(userId: string) {
    return this.prisma.mLFeatureVector.findUnique({
      where: { userId },
    });
  }
}
```

### 2. Create Matching Module (`apps/api/src/matching/`)

**matching.module.ts**
```typescript
import { Module } from '@nestjs/common';
import { MatchingService } from './matching.service';
import { MatchingController } from './matching.controller';
import { QuizModule } from '../quiz/quiz.module';

@Module({
  imports: [QuizModule],
  providers: [MatchingService],
  controllers: [MatchingController],
})
export class MatchingModule {}
```

**matching.controller.ts**
```typescript
import { Controller, Get, Post, Body, Query, UseGuards } from '@nestjs/common';
import { MatchingService } from './matching.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { User } from '../decorators/user.decorator';

@Controller('matches')
@UseGuards(JwtAuthGuard)
export class MatchingController {
  constructor(private matchingService: MatchingService) {}

  @Get('suggested')
  async getSuggestedMatches(
    @User() user: any,
    @Query('limit') limit: string = '20',
  ) {
    return this.matchingService.getSuggestedMatches(user.id, parseInt(limit));
  }

  @Post('interact')
  async recordInteraction(
    @User() user: any,
    @Body() data: { targetUserId: string; action: 'like' | 'skip' | 'super_like' },
  ) {
    return this.matchingService.recordInteraction(
      user.id,
      data.targetUserId,
      data.action,
    );
  }
}
```

**matching.service.ts**
```typescript
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { QuizService } from '../quiz/quiz.service';

@Injectable()
export class MatchingService {
  constructor(
    private prisma: PrismaService,
    private quizService: QuizService,
  ) {}

  async getSuggestedMatches(userId: string, limit: number = 20) {
    // 1. Get user's feature vector
    const userVector = await this.quizService.getUserFeatureVector(userId);
    if (!userVector) {
      throw new Error('User has not completed quiz');
    }

    // 2. Get all other users with feature vectors
    const candidates = await this.prisma.mLFeatureVector.findMany({
      where: {
        userId: { not: userId },
      },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
            bio: true,
          },
        },
      },
    });

    // 3. Calculate compatibility scores
    const scoredCandidates = candidates.map((candidate) => {
      const score = this.calculateCompatibility(
        userVector.features as any,
        candidate.features as any,
      );
      return {
        user: candidate.user,
        compatibilityScore: score,
      };
    });

    // 4. Sort and limit
    return scoredCandidates
      .sort((a, b) => b.compatibilityScore.overallScore - a.compatibilityScore.overallScore)
      .slice(0, limit);
  }

  async recordInteraction(
    userId: string,
    targetUserId: string,
    action: 'like' | 'skip' | 'super_like',
  ) {
    // 1. Get feature vectors
    const userVector = await this.quizService.getUserFeatureVector(userId);
    const targetVector = await this.quizService.getUserFeatureVector(targetUserId);

    // 2. Calculate compatibility
    const compatibilityScore = userVector && targetVector
      ? this.calculateCompatibility(userVector.features as any, targetVector.features as any)
      : null;

    // 3. Check if other user liked this user (mutual match)
    const existingInteraction = await this.prisma.userInteraction.findFirst({
      where: {
        userId: targetUserId,
        targetUserId: userId,
        action: { in: ['like', 'super_like'] },
      },
    });

    const matched = action === 'like' || action === 'super_like'
      ? !!existingInteraction
      : false;

    // 4. Record interaction
    const interaction = await this.prisma.userInteraction.create({
      data: {
        userId,
        targetUserId,
        action,
        matched,
        compatibilityScore: compatibilityScore?.overallScore,
      },
    });

    return {
      interactionId: interaction.id,
      matched,
      compatibilityScore,
    };
  }

  // Copy the calculateCompatibility function from frontend
  // apps/web/src/lib/ml/compatibilityScorer.ts
  private calculateCompatibility(userFeatures: any, candidateFeatures: any) {
    // TODO: Implement this - copy from frontend compatibilityScorer.ts
    // For now, return a simple score
    return {
      overallScore: 75,
      categoryScores: {
        petPersonality: 80,
        activityLevel: 70,
        socialization: 75,
        lifestyleMatch: 75,
      },
      insights: ['Great potential match!'],
    };
  }
}
```

### 3. Create ML Module (`apps/api/src/ml/`)

**ml.module.ts** - For training data collection
```typescript
import { Module } from '@nestjs/common';
import { MLService } from './ml.service';
import { MLController } from './ml.controller';

@Module({
  providers: [MLService],
  controllers: [MLController],
})
export class MLModule {}
```

**ml.controller.ts**
```typescript
import { Controller, Post, Body, UseGuards } from '@nestjs/common';
import { MLService } from './ml.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

@Controller('ml')
@UseGuards(JwtAuthGuard)
export class MLController {
  constructor(private mlService: MLService) {}

  @Post('training-data')
  async syncTrainingData(@Body() data: { dataPoints: any[] }) {
    return this.mlService.storeTrainingData(data.dataPoints);
  }
}
```

**ml.service.ts**
```typescript
import { Injectable } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';

@Injectable()
export class MLService {
  constructor(private prisma: PrismaService) {}

  async storeTrainingData(dataPoints: any[]) {
    const stored = await this.prisma.mLTrainingData.createMany({
      data: dataPoints.map((point) => ({
        dataPoint: point,
        label: this.calculateLabel(point),
      })),
    });

    return {
      stored: stored.count,
      totalDataPoints: await this.prisma.mLTrainingData.count(),
    };
  }

  private calculateLabel(point: any): number {
    if (point.meetupRating) {
      return point.meetupRating / 5; // Normalize to 0-1
    } else if (point.matched) {
      return 0.8;
    } else if (point.userLiked) {
      return 0.6;
    }
    return 0;
  }
}
```

### 4. Update app.module.ts

Add the new modules to `apps/api/src/app.module.ts`:

```typescript
import { QuizModule } from './quiz/quiz.module';
import { MatchingModule } from './matching/matching.module';
import { MLModule } from './ml/ml.module';

@Module({
  imports: [
    // ... existing modules
    QuizModule,
    MatchingModule,
    MLModule,
  ],
  // ...
})
export class AppModule {}
```

### 5. Copy Compatibility Algorithm

Copy the compatibility scoring algorithm from the frontend to the backend:
- Source: `apps/web/src/lib/ml/compatibilityScorer.ts`
- Create: `apps/api/src/matching/compatibility-scorer.ts`
- Import and use in `MatchingService`

This ensures the same algorithm runs on both frontend (for preview) and backend (for official matching).

### 6. Testing

Create test API calls:

```bash
# 1. Submit Quiz
curl -X POST http://localhost:4000/api/v1/quiz/submit \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session": {
      "id": "quiz_123",
      "userId": "user_id",
      "responses": [...],
      "completedAt": "2025-01-05T00:00:00Z"
    },
    "featureVector": {
      "userId": "user_id",
      "features": {...}
    }
  }'

# 2. Get Matches
curl -X GET "http://localhost:4000/api/v1/matches/suggested?limit=10" \
  -H "Authorization: Bearer $TOKEN"

# 3. Record Interaction
curl -X POST http://localhost:4000/api/v1/matches/interact \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "targetUserId": "target_user_id",
    "action": "like"
  }'

# 4. Sync Training Data
curl -X POST http://localhost:4000/api/v1/ml/training-data \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataPoints": [...]
  }'
```

## Summary

**Database**: ✅ Complete (Prisma schema + migration)
**Backend Modules**: 🚧 Need to create (quiz, matching, ml)
**Algorithm**: ⏳ Need to port from frontend to backend
**Testing**: ⏳ Need to test end-to-end flow

Once the backend is complete, the full quiz → matching → interaction flow will work!
