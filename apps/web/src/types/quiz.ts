export type QuizAnswer = string | string[] | number;

export interface QuizOption {
  id: string;
  label: string;
  value: string | number;
  description?: string;
}

export interface QuizQuestion {
  id: string;
  sectionId: string;
  question: string;
  description?: string;
  type: 'multiple_choice' | 'multiple_select' | 'scale' | 'text';
  options?: QuizOption[];
  required?: boolean;
  allowCustom?: boolean;
  scaleRange?: {
    min: number;
    max: number;
    minLabel: string;
    maxLabel: string;
  };
}

export interface QuizResponse {
  questionId: string;
  answer: QuizAnswer;
  customAnswer?: string;
  timestamp: string;
}

export interface QuizSession {
  id: string;
  userId: string;
  petId?: string;
  responses: QuizResponse[];
  completedAt: string;
  currentStep: number;
  totalSteps: number;
}

export interface MLFeatureVector {
  userId: string;
  petId?: string;
  timestamp: string;
  features: {
    energyLevel: number;
    socialability: number;
    trainingLevel: number;
    playStyle: string[];
    preferredActivities: string[];
    activityFrequency: number;
    experienceLevel: number;
    availableTimePerDay: number;
    preferredTimes: string[];
    distanceWillingness: number;
    groupSizePreference: string;
    environmentPreference: string[];
  };
}

export interface CompatibilityScore {
  overallScore: number;
  categoryScores: {
    petPersonality: number;
    activityLevel: number;
    socialization: number;
    lifestyleMatch: number;
  };
  insights: string[];
}

export interface MLTrainingData {
  features: number[][];
  labels: number[];
  featureNames: string[];
}
