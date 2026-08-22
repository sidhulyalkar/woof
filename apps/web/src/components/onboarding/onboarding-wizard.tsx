'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Progress } from '@/components/ui/progress';
import { UserProfileStep } from './steps/user-profile-step';
import { PetProfileStep } from './steps/pet-profile-step';
import { PreferencesStep } from './steps/preferences-step';
import { PermissionsStep } from './steps/permissions-step';
import type { OnboardingData } from './onboarding-types';

const steps = [
  { id: 'profile', title: 'Your Profile', component: UserProfileStep },
  { id: 'pet', title: 'Pet Profile', component: PetProfileStep },
  { id: 'preferences', title: 'Preferences', component: PreferencesStep },
  { id: 'permissions', title: 'Permissions', component: PermissionsStep },
];

export function OnboardingWizard() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<OnboardingData>({});

  const progress = ((currentStep + 1) / steps.length) * 100;
  const CurrentStepComponent = steps[currentStep].component;

  const handleNext = (data: Partial<OnboardingData>) => {
    const nextData = { ...formData, ...data };
    setFormData(nextData);

    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      void handleComplete(nextData);
    }
  };

  const handleBack = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleComplete = async (data: OnboardingData) => {
    try {
      // This legacy wizard currently hands off only local preferences. Durable account/pet
      // creation is owned by the canonical onboarding route, so do not imply persistence here.
      console.info('Onboarding flow completed', {
        hasPetProfile: Boolean(data.petName),
        preferenceCount: data.activityPreferences?.length ?? 0,
      });
      router.push('/');
    } catch (error) {
      console.error('Onboarding failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="mx-auto max-w-2xl space-y-8 py-8">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              Step {currentStep + 1} of {steps.length}
            </span>
            <span className="font-medium">{steps[currentStep].title}</span>
          </div>
          <Progress value={progress} />
        </div>

        <CurrentStepComponent
          data={formData}
          onNext={handleNext}
          onBack={currentStep > 0 ? handleBack : undefined}
        />
      </div>
    </div>
  );
}
