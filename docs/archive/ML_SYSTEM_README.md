# PetPath ML Matching System

## Overview

The PetPath ML matching system is designed to provide intelligent pet-to-pet matching based on comprehensive behavioral and lifestyle data collected through an onboarding quiz. The system is built with a hybrid approach:

1. **Phase 1 (Current)**: Rule-based compatibility scoring with ML data collection
2. **Phase 2 (Future)**: Machine learning model trained on real user interaction data

## System Architecture

### 1. Quiz System (`/src/types/quiz.ts`, `/src/data/quizQuestions.ts`)

**15 comprehensive questions across 5 categories:**

- **Pet Personality** (3 questions)
  - Energy level (1-10 scale)
  - Temperament (multiple choice + custom)
  - Play style (multiple select + custom)

- **Socialization** (3 questions)
  - Comfort with other dogs (1-10 scale)
  - Comfort with people (1-10 scale)
  - Preferred group size

- **Activity Level** (3 questions)
  - Walk frequency
  - Walk duration
  - Preferred activities (multiple select + custom)

- **Owner Lifestyle** (3 questions)
  - Preferred walking times (multiple select)
  - Experience level
  - Travel distance willing

- **Preferences** (3 questions)
  - Environment preferences (multiple select + custom)
  - Training interest (1-10 scale)
  - Meetup goals (multiple select + custom)

**ML Weight System**: Each question has an `mlWeight` value (0.0-1.0) indicating its importance for matching:
- Critical factors (energy, socialization): 0.9-0.95
- Important factors (activity, play style): 0.8-0.85
- Secondary factors (schedule, environment): 0.65-0.75
- Nice-to-have factors (training interest): 0.5-0.6

### 2. Feature Vector Generation (`/src/lib/ml/compatibilityScorer.ts`)

Quiz responses are transformed into structured ML features:

```typescript
MLFeatureVector {
  userId: string
  petId: string
  features: {
    // Pet characteristics (normalized 1-10)
    energyLevel: number
    socialability: number
    trainingLevel: number
    playStyle: string[]
    preferredActivities: string[]

    // Owner characteristics
    activityFrequency: number        // walks per week
    experienceLevel: number          // 1-5 scale
    availableTimePerDay: number      // hours
    preferredTimes: string[]

    // Preferences
    distanceWillingness: number      // km
    groupSizePreference: string
    environmentPreference: string[]
  }
}
```

### 3. Compatibility Scoring Algorithm

**Current Implementation** (`calculateCompatibility()`):

Weighted scoring across 7 dimensions:

| Dimension | Weight | Calculation Method |
|-----------|--------|-------------------|
| Energy Level Match | 25% | Inverse of absolute difference (0-10 scale) |
| Socialability Match | 20% | Inverse of absolute difference (0-10 scale) |
| Activity Level Match | 15% | Comparison of walk frequency |
| Play Style Match | 15% | Jaccard similarity of play style arrays |
| Schedule Match | 10% | Jaccard similarity of preferred times |
| Environment Match | 10% | Jaccard similarity of environment preferences |
| Group Size Match | 5% | Compatibility matrix lookup |

**Formula**:
```
Overall Score (0-100) = Σ(dimension_score × weight)
```

**Insight Generation**:
- Automatic generation of human-readable compatibility insights
- Highlights both strengths and potential challenges
- Examples:
  - "🔋 Perfect energy level match - your pets will have a blast together!"
  - "📅 Schedule alignment - you both prefer similar walking times"
  - "⚡ Different energy levels - may need supervised introductions"

### 4. Training Data Collection (`/src/lib/ml/trainingDataCollector.ts`)

**Purpose**: Collect real user interaction data to train future ML models

**Data Collected**:
- User interactions (like, skip, super_like)
- Successful matches
- Meetup completions and ratings (1-5 stars)

**Storage**:
- Local browser storage (up to 1000 data points)
- Exportable to CSV or JSON
- Backend sync capability (for production)

**Training Data Format**:
```typescript
MLTrainingDataPoint {
  userFeatures: MLFeatureVector
  candidateFeatures: MLFeatureVector

  // Calculated features
  energyDifference: number
  socialDifference: number
  playStyleOverlap: number
  scheduleOverlap: number

  // Outcome labels (collected from user behavior)
  userLiked?: boolean
  matched?: boolean
  meetupCompleted?: boolean
  meetupRating?: number  // 1-5 stars
}
```

### 5. Feature Engineering for ML

**Matrix Generation** (`FeatureEngineer.createFeatureMatrix()`):

13 numerical features extracted from each interaction:
1. Energy difference
2. Social difference
3. Play style overlap (Jaccard)
4. Schedule overlap (Jaccard)
5-6. User & candidate energy levels
7-8. User & candidate socialability scores
9-10. User & candidate activity frequency
11-12. User & candidate experience levels
13. Distance willingness product

**Label Generation**:
Normalized 0-1 scale based on interaction outcome:
- Meetup rating (5 stars) → 1.0
- Meetup rating (1 star) → 0.2
- Match (no meetup yet) → 0.8
- Like (no match yet) → 0.6
- Skip/no interaction → 0.0

## Integration Points

### Frontend Integration

1. **Onboarding Flow**: `/onboarding/quiz` route
2. **Quiz Components**: `OnboardingQuiz` and `QuizQuestionCard`
3. **API Hooks**: `useSubmitQuiz`, `useGetMatches`, `useRecordInteraction`

### Backend Requirements (TODO)

Create the following API endpoints:

```typescript
POST /api/quiz/submit
Body: { session: QuizSession, featureVector: MLFeatureVector }
Response: { success: boolean, userId: string }

GET /api/matches/suggested
Response: { matches: Array<{ user, pet, compatibilityScore }> }

POST /api/matches/interact
Body: { targetUserId: string, action: 'like' | 'skip' | 'super_like' }
Response: { matched: boolean }

POST /api/ml/training-data
Body: { dataPoints: MLTrainingDataPoint[] }
Response: { stored: number }
```

## Future ML Model Training

### Phase 2 Implementation Plan

**Step 1: Data Collection** (Current MVP - Beta Launch)
- Collect 500-1000 user interactions minimum
- Track conversion funnel: views → likes → matches → meetups → ratings
- Target metrics:
  - 100+ beta users
  - 1000+ interaction data points
  - 50+ completed meetups with ratings

**Step 2: Model Training** (Post-Beta)

Recommended approach: **Gradient Boosting (XGBoost/LightGBM)**

Why:
- Excellent for tabular data with mixed feature types
- Handles non-linear relationships well
- Less data required than deep learning
- Interpretable feature importance

Alternative: **Neural Network** (if dataset grows large)
- Use TensorFlow.js for in-browser inference
- Train Python model, export to TF.js format

**Step 3: Model Evaluation**
- Split data: 70% train, 15% validation, 15% test
- Metrics:
  - Precision@k for match recommendations
  - NDCG (Normalized Discounted Cumulative Gain)
  - Correlation between predicted score and actual meetup ratings

**Step 4: A/B Testing**
- 50% users: Rule-based algorithm (control)
- 50% users: ML model (treatment)
- Compare:
  - Match acceptance rate
  - Meetup completion rate
  - Average meetup ratings

### Python Training Script Template

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load exported CSV data
data = pd.read_csv('training_data.csv')

# Features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Train model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Features:")
print(feature_importance.head(10))
```

## Analytics & Metrics

### Quiz Completion Metrics
- Completion rate
- Average time to complete
- Question skip rate
- Custom answer usage rate

### Matching Metrics
- Like rate by compatibility score bucket
- Match rate (mutual likes)
- Meetup completion rate
- Average meetup rating
- User satisfaction scores

### ML Performance Metrics
- Training data collection rate
- Model prediction accuracy
- A/B test conversion metrics
- Feature importance rankings

## Data Export for Investors

Use `trainingDataCollector.exportTrainingData()` to generate investor-ready reports:

```javascript
const report = trainingDataCollector.exportTrainingData();

console.log(`
Total User Interactions: ${report.metadata.totalSamples}
Like Rate: ${(report.metadata.likeCount / report.metadata.totalSamples * 100).toFixed(1)}%
Match Rate: ${(report.metadata.matchCount / report.metadata.totalSamples * 100).toFixed(1)}%
Meetup Completion: ${report.metadata.meetupCount}
Average Meetup Rating: ${report.metadata.avgMeetupRating}/5.0
`);
```

## Best Practices

### Quiz Design
✅ Keep questions under 15 total
✅ Allow custom answers for flexibility
✅ Use 1-10 scales for subjective measures
✅ Provide clear, friendly question text
✅ Show progress bar for motivation

### Data Collection
✅ Store training data locally AND sync to backend
✅ Anonymize sensitive user data
✅ Get user consent for ML training
✅ Provide data export/deletion options

### Matching Algorithm
✅ Weight critical factors (energy, sociability) highest
✅ Generate human-readable insights
✅ Show confidence scores to users
✅ Allow users to adjust preferences
✅ Continuously A/B test improvements

## Technical Stack

**Current**:
- TypeScript for type safety
- React Query for data fetching
- localStorage for training data persistence
- Rule-based algorithm for immediate matching

**Future** (Phase 2):
- TensorFlow.js or ONNX.js for in-browser inference
- Python (scikit-learn/XGBoost) for model training
- PostgreSQL for storing training data
- Airflow/n8n for automated retraining pipeline

## License & Attribution

This ML system is part of the PetPath MVP and is proprietary to PetPath Inc.

---

**Last Updated**: 2025-01-XX
**Version**: 1.0.0 (Beta)
**Contact**: dev@petpath.com
