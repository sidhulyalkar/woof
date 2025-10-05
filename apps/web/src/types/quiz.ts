/**
 * PetPath Onboarding Quiz Types
 * Comprehensive quiz system for pet behavior and owner preferences matching
 */

export type QuizQuestionType = 'multiple_choice' | 'multiple_select' | 'scale' | 'text';

export interface QuizOption {
  id: string;
  label: string;
  value: string | number;
  description?: string;
  /** Weight for ML scoring algorithm */
  weight?: number;
}

export interface QuizQuestion {
  id: string;
  category: QuizCategory;
  type: QuizQuestionType;
  question: string;
  description?: string;
  required: boolean;
  options?: QuizOption[];
  /** Allow custom text input alongside options */
  allowCustom?: boolean;
  /** For scale questions */
  scaleRange?: { min: number; max: number; minLabel: string; maxLabel: string };
  /** ML feature importance for matching algorithm */
  mlWeight: number;
}

export type QuizCategory =
  | 'pet_personality'
  | 'pet_behavior'
  | 'activity_level'
  | 'socialization'
  | 'owner_lifestyle'
  | 'preferences';

export interface QuizResponse {
  questionId: string;
  answer: string | string[] | number;
  customAnswer?: string;
  timestamp: string;
}

export interface QuizSession {
  id: string;
  userId: string;
  petId?: string;
  responses: QuizResponse[];
  completedAt?: string;
  currentStep: number;
  totalSteps: number;
}

export interface CompatibilityScore {
  overallScore: number; // 0-100
  categoryScores: {
    petPersonality: number;
    activityLevel: number;
    socialization: number;
    lifestyleMatch: number;
  };
  insights: string[];
  topMatches?: string[]; // Pet IDs
}

export interface MLFeatureVector {
  userId: string;
  petId?: string;
  features: {
    // Pet characteristics
    energyLevel: number; // 1-10
    socialability: number; // 1-10
    trainingLevel: number; // 1-10
    playStyle: string[];
    preferredActivities: string[];

    // Owner characteristics
    activityFrequency: number; // times per week
    experienceLevel: number; // 1-10
    availableTimePerDay: number; // hours
    preferredTimes: string[]; // morning, afternoon, evening

    // Preferences
    distanceWillingness: number; // km
    groupSizePreference: string; // solo, small, large
    environmentPreference: string[]; // park, trail, beach, etc.
  };
  timestamp: string;
}

export interface MLTrainingData {
  sessionId: string;
  userFeatures: MLFeatureVector;
  interactions: {
    likedPetIds: string[];
    matchedPetIds: string[];
    meetupCompleted: string[];
    meetupRatings: { petId: string; rating: number }[];
  };
  timestamp: string;
}
