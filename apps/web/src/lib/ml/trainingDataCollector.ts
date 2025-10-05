/**
 * ML Training Data Collection Pipeline
 * Collects user interaction data for future ML model training
 */

import { MLTrainingDataPoint, prepareTrainingDataPoint } from './compatibilityScorer';
import { MLFeatureVector, MLTrainingData } from '@/types/quiz';

/**
 * Training Data Collector
 * Stores interaction data for ML model training
 */
export class TrainingDataCollector {
  private storageKey = 'woof_ml_training_data';
  private maxStorageSize = 1000; // Maximum number of data points to store locally

  /**
   * Record a user interaction (like/skip)
   */
  recordInteraction(
    userVector: MLFeatureVector,
    candidateVector: MLFeatureVector,
    interaction: 'like' | 'skip' | 'super_like'
  ): void {
    const dataPoint = prepareTrainingDataPoint(userVector, candidateVector, {
      userLiked: interaction === 'like' || interaction === 'super_like',
    });

    this.storeDataPoint(dataPoint);
  }

  /**
   * Record a match event
   */
  recordMatch(
    userVector: MLFeatureVector,
    candidateVector: MLFeatureVector
  ): void {
    const dataPoint = prepareTrainingDataPoint(userVector, candidateVector, {
      userLiked: true,
      matched: true,
    });

    this.storeDataPoint(dataPoint);
  }

  /**
   * Record a meetup completion and rating
   */
  recordMeetup(
    userVector: MLFeatureVector,
    candidateVector: MLFeatureVector,
    rating: number
  ): void {
    const dataPoint = prepareTrainingDataPoint(userVector, candidateVector, {
      userLiked: true,
      matched: true,
      meetupCompleted: true,
      meetupRating: rating,
    });

    this.storeDataPoint(dataPoint);
  }

  /**
   * Store a data point locally
   */
  private storeDataPoint(dataPoint: MLTrainingDataPoint): void {
    try {
      const existing = this.getStoredData();
      existing.push(dataPoint);

      // Keep only the most recent data points
      if (existing.length > this.maxStorageSize) {
        existing.shift();
      }

      localStorage.setItem(this.storageKey, JSON.stringify(existing));
    } catch (error) {
      console.error('Failed to store training data:', error);
    }
  }

  /**
   * Get all stored training data
   */
  getStoredData(): MLTrainingDataPoint[] {
    try {
      const data = localStorage.getItem(this.storageKey);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Failed to retrieve training data:', error);
      return [];
    }
  }

  /**
   * Export training data for analysis
   * Returns data in a format ready for ML model training
   */
  exportTrainingData(): {
    metadata: {
      totalSamples: number;
      likeCount: number;
      matchCount: number;
      meetupCount: number;
      avgMeetupRating: number;
      exportDate: string;
    };
    data: MLTrainingDataPoint[];
  } {
    const data = this.getStoredData();

    const likeCount = data.filter((d) => d.userLiked).length;
    const matchCount = data.filter((d) => d.matched).length;
    const meetupCount = data.filter((d) => d.meetupCompleted).length;
    const avgMeetupRating =
      meetupCount > 0
        ? data
            .filter((d) => d.meetupRating !== undefined)
            .reduce((sum, d) => sum + (d.meetupRating || 0), 0) / meetupCount
        : 0;

    return {
      metadata: {
        totalSamples: data.length,
        likeCount,
        matchCount,
        meetupCount,
        avgMeetupRating: Math.round(avgMeetupRating * 100) / 100,
        exportDate: new Date().toISOString(),
      },
      data,
    };
  }

  /**
   * Clear stored training data
   */
  clearData(): void {
    try {
      localStorage.removeItem(this.storageKey);
    } catch (error) {
      console.error('Failed to clear training data:', error);
    }
  }

  /**
   * Get analytics about stored data
   */
  getAnalytics(): {
    totalInteractions: number;
    likeRate: number;
    matchRate: number;
    meetupCompletionRate: number;
    avgMeetupRating: number;
  } {
    const data = this.getStoredData();
    const totalInteractions = data.length;

    if (totalInteractions === 0) {
      return {
        totalInteractions: 0,
        likeRate: 0,
        matchRate: 0,
        meetupCompletionRate: 0,
        avgMeetupRating: 0,
      };
    }

    const likeCount = data.filter((d) => d.userLiked).length;
    const matchCount = data.filter((d) => d.matched).length;
    const meetupCount = data.filter((d) => d.meetupCompleted).length;
    const ratingsSum = data
      .filter((d) => d.meetupRating !== undefined)
      .reduce((sum, d) => sum + (d.meetupRating || 0), 0);

    return {
      totalInteractions,
      likeRate: (likeCount / totalInteractions) * 100,
      matchRate: (matchCount / totalInteractions) * 100,
      meetupCompletionRate: matchCount > 0 ? (meetupCount / matchCount) * 100 : 0,
      avgMeetupRating: meetupCount > 0 ? ratingsSum / meetupCount : 0,
    };
  }
}

/**
 * Feature Engineering for ML Models
 * Transforms raw data into features suitable for machine learning
 */
export class FeatureEngineer {
  /**
   * Create feature matrix from training data
   * This format can be used with TensorFlow.js or exported for Python ML libraries
   */
  createFeatureMatrix(dataPoints: MLTrainingDataPoint[]): {
    features: number[][];
    labels: number[];
    featureNames: string[];
  } {
    const featureNames = [
      'energyDifference',
      'socialDifference',
      'playStyleOverlap',
      'scheduleOverlap',
      'userEnergyLevel',
      'candidateEnergyLevel',
      'userSocialability',
      'candidateSocialability',
      'userActivityFrequency',
      'candidateActivityFrequency',
      'userExperienceLevel',
      'candidateExperienceLevel',
      'distanceWillingnessProduct',
    ];

    const features: number[][] = [];
    const labels: number[] = [];

    dataPoints.forEach((point) => {
      // Skip if no outcome label
      if (point.meetupRating === undefined && point.matched === undefined && point.userLiked === undefined) {
        return;
      }

      // Extract features
      const featureVector = [
        point.energyDifference,
        point.socialDifference,
        point.playStyleOverlap,
        point.scheduleOverlap,
        point.userFeatures.features.energyLevel,
        point.candidateFeatures.features.energyLevel,
        point.userFeatures.features.socialability,
        point.candidateFeatures.features.socialability,
        point.userFeatures.features.activityFrequency,
        point.candidateFeatures.features.activityFrequency,
        point.userFeatures.features.experienceLevel,
        point.candidateFeatures.features.experienceLevel,
        point.userFeatures.features.distanceWillingness * point.candidateFeatures.features.distanceWillingness,
      ];

      features.push(featureVector);

      // Create label (priority: meetup rating > matched > liked)
      let label = 0;
      if (point.meetupRating !== undefined) {
        label = point.meetupRating / 5; // Normalize to 0-1
      } else if (point.matched) {
        label = 0.8;
      } else if (point.userLiked) {
        label = 0.6;
      }

      labels.push(label);
    });

    return { features, labels, featureNames };
  }

  /**
   * Export data in CSV format for Python ML tools
   */
  exportToCSV(dataPoints: MLTrainingDataPoint[]): string {
    const { features, labels, featureNames } = this.createFeatureMatrix(dataPoints);

    // Create CSV header
    const header = [...featureNames, 'label'].join(',');

    // Create CSV rows
    const rows = features.map((featureVector, index) => {
      return [...featureVector, labels[index]].join(',');
    });

    return [header, ...rows].join('\n');
  }
}

// Export singleton instances
export const trainingDataCollector = new TrainingDataCollector();
export const featureEngineer = new FeatureEngineer();
