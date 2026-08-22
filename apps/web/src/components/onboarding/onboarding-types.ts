export interface OnboardingData {
  name?: string;
  bio?: string;
  location?: string;
  petName?: string;
  petSpecies?: string;
  petBreed?: string;
  activityPreferences?: string[];
  allowNotifications?: boolean;
  allowLocation?: boolean;
}

export interface OnboardingStepProps {
  data: OnboardingData;
  onNext: (data: Partial<OnboardingData>) => void;
  onBack?: () => void;
}
