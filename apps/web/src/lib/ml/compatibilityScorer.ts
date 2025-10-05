/**
 * ML Compatibility Scoring Framework
 * Initial rule-based algorithm with hooks for future ML model integration
 */

import { QuizSession, CompatibilityScore, MLFeatureVector } from '@/types/quiz';
import { QUIZ_QUESTIONS } from '@/data/quizQuestions';

/**
 * Convert quiz responses to ML feature vector
 */
export function quizToFeatureVector(session: QuizSession): MLFeatureVector {
  const responses = session.responses;

  // Helper to get response value
  const getResponse = (questionId: string) => {
    return responses.find((r) => r.questionId === questionId)?.answer;
  };

  // Helper to get numeric value
  const getNumeric = (questionId: string, defaultValue = 5): number => {
    const value = getResponse(questionId);
    return typeof value === 'number' ? value : typeof value === 'string' ? parseFloat(value) || defaultValue : defaultValue;
  };

  // Helper to get array value
  const getArray = (questionId: string): string[] => {
    const value = getResponse(questionId);
    return Array.isArray(value) ? value : value ? [value.toString()] : [];
  };

  return {
    userId: session.userId,
    petId: session.petId,
    features: {
      // Pet characteristics
      energyLevel: getNumeric('pet_energy_level', 5),
      socialability: getNumeric('pet_dog_socialization', 5),
      trainingLevel: getNumeric('training_interest', 5),
      playStyle: getArray('pet_play_style'),
      preferredActivities: getArray('preferred_activities'),

      // Owner characteristics
      activityFrequency: getNumeric('walk_frequency', 1),
      experienceLevel: getNumeric('experience_level', 2),
      availableTimePerDay: getNumeric('walk_duration', 30) / 60, // Convert minutes to hours
      preferredTimes: getArray('preferred_times'),

      // Preferences
      distanceWillingness: getNumeric('travel_distance', 2),
      groupSizePreference: (getResponse('preferred_group_size') as string) || 'small',
      environmentPreference: getArray('environment_preference'),
    },
    timestamp: new Date().toISOString(),
  };
}

/**
 * Calculate compatibility score between two feature vectors
 * Returns a score from 0-100
 */
export function calculateCompatibility(userVector: MLFeatureVector, candidateVector: MLFeatureVector): CompatibilityScore {
  const weights = {
    energyLevel: 0.25,
    socialability: 0.20,
    activityMatch: 0.15,
    playStyleMatch: 0.15,
    scheduleMatch: 0.10,
    environmentMatch: 0.10,
    groupSizeMatch: 0.05,
  };

  // 1. Energy Level Match (0-100)
  const energyDiff = Math.abs(userVector.features.energyLevel - candidateVector.features.energyLevel);
  const energyScore = Math.max(0, 100 - energyDiff * 10);

  // 2. Socialability Match (0-100)
  const socialDiff = Math.abs(userVector.features.socialability - candidateVector.features.socialability);
  const socialScore = Math.max(0, 100 - socialDiff * 10);

  // 3. Activity Level Match (0-100)
  const activityDiff = Math.abs(userVector.features.activityFrequency - candidateVector.features.activityFrequency);
  const activityScore = Math.max(0, 100 - activityDiff * 15);

  // 4. Play Style Match (0-100) - Jaccard similarity
  const playStyleScore = calculateJaccardSimilarity(
    userVector.features.playStyle,
    candidateVector.features.playStyle
  ) * 100;

  // 5. Schedule Match (0-100) - Time overlap
  const scheduleScore = calculateJaccardSimilarity(
    userVector.features.preferredTimes,
    candidateVector.features.preferredTimes
  ) * 100;

  // 6. Environment Match (0-100)
  const environmentScore = calculateJaccardSimilarity(
    userVector.features.environmentPreference,
    candidateVector.features.environmentPreference
  ) * 100;

  // 7. Group Size Match (0-100) - Exact match or compatible
  const groupSizeScore = calculateGroupSizeCompatibility(
    userVector.features.groupSizePreference,
    candidateVector.features.groupSizePreference
  );

  // Calculate weighted overall score
  const overallScore = Math.round(
    energyScore * weights.energyLevel +
    socialScore * weights.socialability +
    activityScore * weights.activityMatch +
    playStyleScore * weights.playStyleMatch +
    scheduleScore * weights.scheduleMatch +
    environmentScore * weights.environmentMatch +
    groupSizeScore * weights.groupSizeMatch
  );

  // Generate insights
  const insights = generateInsights({
    energyScore,
    socialScore,
    activityScore,
    playStyleScore,
    scheduleScore,
    environmentScore,
    groupSizeScore,
  });

  return {
    overallScore,
    categoryScores: {
      petPersonality: Math.round((energyScore + socialScore) / 2),
      activityLevel: Math.round(activityScore),
      socialization: Math.round(socialScore),
      lifestyleMatch: Math.round((scheduleScore + environmentScore) / 2),
    },
    insights,
  };
}

/**
 * Jaccard Similarity for sets
 */
function calculateJaccardSimilarity(set1: string[], set2: string[]): number {
  if (set1.length === 0 && set2.length === 0) return 1;
  if (set1.length === 0 || set2.length === 0) return 0;

  const intersection = set1.filter((item) => set2.includes(item)).length;
  const union = new Set([...set1, ...set2]).size;

  return intersection / union;
}

/**
 * Group size compatibility logic
 */
function calculateGroupSizeCompatibility(size1: string, size2: string): number {
  const compatibilityMatrix: Record<string, Record<string, number>> = {
    one_on_one: { one_on_one: 100, small: 80, medium: 40, large: 20, varies: 60 },
    small: { one_on_one: 80, small: 100, medium: 70, large: 50, varies: 80 },
    medium: { one_on_one: 40, small: 70, medium: 100, large: 80, varies: 80 },
    large: { one_on_one: 20, small: 50, medium: 80, large: 100, varies: 70 },
    varies: { one_on_one: 60, small: 80, medium: 80, large: 70, varies: 100 },
  };

  return compatibilityMatrix[size1]?.[size2] || 50;
}

/**
 * Generate human-readable insights
 */
function generateInsights(scores: {
  energyScore: number;
  socialScore: number;
  activityScore: number;
  playStyleScore: number;
  scheduleScore: number;
  environmentScore: number;
  groupSizeScore: number;
}): string[] {
  const insights: string[] = [];

  if (scores.energyScore > 80) {
    insights.push('🔋 Perfect energy level match - your pets will have a blast together!');
  } else if (scores.energyScore < 50) {
    insights.push('⚡ Different energy levels - may need supervised introductions');
  }

  if (scores.socialScore > 80) {
    insights.push('🤝 Great social compatibility - both pets are equally comfortable with others');
  }

  if (scores.activityScore > 80) {
    insights.push('🏃 Excellent activity match - similar exercise routines');
  }

  if (scores.playStyleScore > 70) {
    insights.push('🎾 Compatible play styles - they enjoy similar activities');
  }

  if (scores.scheduleScore > 70) {
    insights.push('📅 Schedule alignment - you both prefer similar walking times');
  } else if (scores.scheduleScore < 40) {
    insights.push('⏰ Different schedules - coordination may be needed');
  }

  if (scores.environmentScore > 70) {
    insights.push('🌳 Love the same spots - enjoy similar environments');
  }

  if (insights.length === 0) {
    insights.push('👍 Good potential for a compatible match');
  }

  return insights;
}

/**
 * Batch score multiple candidates
 */
export function scoreCandidates(
  userVector: MLFeatureVector,
  candidates: MLFeatureVector[]
): Array<{ candidateId: string; score: CompatibilityScore }> {
  return candidates
    .map((candidate) => ({
      candidateId: candidate.userId,
      score: calculateCompatibility(userVector, candidate),
    }))
    .sort((a, b) => b.score.overallScore - a.score.overallScore);
}

/**
 * ML Training Data Export Format
 * This structure can be used to train a more sophisticated ML model in the future
 */
export interface MLTrainingDataPoint {
  // Input features
  userFeatures: MLFeatureVector;
  candidateFeatures: MLFeatureVector;

  // Calculated features
  energyDifference: number;
  socialDifference: number;
  playStyleOverlap: number;
  scheduleOverlap: number;

  // Outcome label (to be collected from user interactions)
  userLiked?: boolean;
  matched?: boolean;
  meetupCompleted?: boolean;
  meetupRating?: number; // 1-5 stars

  timestamp: string;
}

/**
 * Prepare data point for ML training
 */
export function prepareTrainingDataPoint(
  userVector: MLFeatureVector,
  candidateVector: MLFeatureVector,
  outcome?: {
    userLiked?: boolean;
    matched?: boolean;
    meetupCompleted?: boolean;
    meetupRating?: number;
  }
): MLTrainingDataPoint {
  return {
    userFeatures: userVector,
    candidateFeatures: candidateVector,
    energyDifference: Math.abs(userVector.features.energyLevel - candidateVector.features.energyLevel),
    socialDifference: Math.abs(userVector.features.socialability - candidateVector.features.socialability),
    playStyleOverlap: calculateJaccardSimilarity(
      userVector.features.playStyle,
      candidateVector.features.playStyle
    ),
    scheduleOverlap: calculateJaccardSimilarity(
      userVector.features.preferredTimes,
      candidateVector.features.preferredTimes
    ),
    ...outcome,
    timestamp: new Date().toISOString(),
  };
}
