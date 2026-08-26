'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { isAxiosError } from 'axios';
import { ChevronLeft, PawPrint } from 'lucide-react';
import { toast } from 'sonner';
import { FirstAdventureStep } from '@/components/onboarding/first-adventure-step';
import { OwnerInfoStep, type OwnerInfoData } from '@/components/onboarding/owner-info-step';
import { PetInfoStep, type PetInfoData } from '@/components/onboarding/pet-info-step';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { authApi, storageApi } from '@/lib/api';
import { adaptiveProfileApi } from '@/lib/api/adaptive-profile';
import { petsApi } from '@/lib/api/pets';
import {
  buildFirstAdventureResponses,
  emptyFirstAdventureSelections,
  type FirstAdventureSelections,
} from '@/lib/onboarding/first-adventure';
import { useAuthStore } from '@/lib/stores/auth-store';

type ApiErrorBody = {
  message?: string | string[];
};

type DurablePair = {
  petId: string;
  householdId: string;
  petName: string;
};

const REGISTRATION_KEY_STORAGE = 'woof:first-adventure:registration-key';
const CREATION_KEY_STORAGE = 'woof:first-adventure:pet-creation-key';
const PAIR_STORAGE = 'woof:first-adventure:durable-pair';

function apiErrorMessage(error: unknown, fallback: string) {
  const responseMessage = isAxiosError<ApiErrorBody>(error)
    ? error.response?.data?.message
    : undefined;
  if (Array.isArray(responseMessage)) return responseMessage.join(' ');
  return responseMessage || fallback;
}

function readStoredPair(): DurablePair | null {
  if (typeof window === 'undefined') return null;
  const raw = window.sessionStorage.getItem(PAIR_STORAGE);
  if (!raw) return null;

  try {
    const value = JSON.parse(raw) as Partial<DurablePair>;
    if (value.petId && value.householdId && value.petName) {
      return {
        petId: value.petId,
        householdId: value.householdId,
        petName: value.petName,
      };
    }
  } catch {
    // A malformed browser-only recovery hint is safe to discard. The server remains canonical.
  }
  window.sessionStorage.removeItem(PAIR_STORAGE);
  return null;
}

export default function OnboardingPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [ownerData, setOwnerData] = useState<OwnerInfoData | null>(null);
  const [petData, setPetData] = useState<PetInfoData | null>(null);
  const [durablePair, setDurablePair] = useState<DurablePair | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const registrationKeyRef = useRef<string | null>(null);
  const creationKeyRef = useRef<string | null>(null);
  const recoveryCheckRef = useRef(false);
  const authUser = useAuthStore((state) => state.user);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  const totalSteps = 3;
  const progress = (currentStep / totalSteps) * 100;

  useEffect(() => {
    let cancelled = false;

    const recover = async () => {
      const storedPair = readStoredPair();
      if (storedPair && isAuthenticated && !recoveryCheckRef.current) {
        recoveryCheckRef.current = true;
        try {
          const profile = await adaptiveProfileApi.getState(
            storedPair.householdId,
            storedPair.petId
          );
          if (cancelled) return;

          if (
            profile.householdId === storedPair.householdId &&
            profile.petId === storedPair.petId
          ) {
            setDurablePair(storedPair);
            setError('');
            setCurrentStep(3);
            return;
          }
        } catch (recoveryError) {
          if (cancelled) return;
          console.warn('Saved onboarding pair could not be re-authorized yet', recoveryError);
          setError(
            'We could not verify the saved pet pair yet. Review the pet details and retry; Woof will reuse the same creation key rather than creating a twin.'
          );
        }
      }

      // Registration may have completed before a refresh or transient pet-create failure.
      // Reuse the authenticated account instead of asking for the password again.
      if (isAuthenticated && authUser && currentStep === 1 && !ownerData && !cancelled) {
        setOwnerData({
          handle: authUser.handle,
          email: authUser.email,
          password: '',
          bio: authUser.bio || '',
        });
        setCurrentStep(2);
      }
    };

    void recover();
    return () => {
      cancelled = true;
    };
  }, [authUser, currentStep, isAuthenticated, ownerData]);

  const getRegistrationKey = () => {
    if (registrationKeyRef.current) return registrationKeyRef.current;

    const stored = window.sessionStorage.getItem(REGISTRATION_KEY_STORAGE);
    if (stored) {
      registrationKeyRef.current = stored;
      return stored;
    }

    const next = crypto.randomUUID();
    window.sessionStorage.setItem(REGISTRATION_KEY_STORAGE, next);
    registrationKeyRef.current = next;
    return next;
  };

  const getCreationKey = () => {
    if (creationKeyRef.current) return creationKeyRef.current;

    const stored = window.sessionStorage.getItem(CREATION_KEY_STORAGE);
    if (stored) {
      creationKeyRef.current = stored;
      return stored;
    }

    const next = `first-adventure:${crypto.randomUUID()}`;
    window.sessionStorage.setItem(CREATION_KEY_STORAGE, next);
    creationKeyRef.current = next;
    return next;
  };

  const clearRegistrationReplayKey = () => {
    window.sessionStorage.removeItem(REGISTRATION_KEY_STORAGE);
    registrationKeyRef.current = null;
  };

  const rememberPair = (pair: DurablePair) => {
    setDurablePair(pair);
    window.sessionStorage.setItem(PAIR_STORAGE, JSON.stringify(pair));
  };

  const clearRecoveryHints = () => {
    clearRegistrationReplayKey();
    window.sessionStorage.removeItem(CREATION_KEY_STORAGE);
    window.sessionStorage.removeItem(PAIR_STORAGE);
    creationKeyRef.current = null;
  };

  const refreshAuthProfile = async () => {
    try {
      const profile = await authApi.me();
      const token = useAuthStore.getState().token;
      if (token) useAuthStore.getState().setAuth(profile, token);
    } catch (refreshError) {
      console.warn('Onboarding profile refresh failed', refreshError);
      toast.warning('Your setup is saved. Woof will refresh the profile again on the next screen.');
    }
  };

  const attachPhoto = async (pair: DurablePair, photoFile: File | null) => {
    if (!photoFile) return;

    try {
      const upload = await storageApi.uploadFile(photoFile, 'pets');
      await petsApi.updatePet(pair.petId, { avatarUrl: upload.url });
    } catch (uploadError) {
      console.warn('Pet photo attachment failed during onboarding', uploadError);
      toast.warning(
        'Your pet is ready, but the photo could not be attached. You can add it later.'
      );
    }
  };

  const handleOwnerComplete = (data: OwnerInfoData) => {
    setOwnerData(data);
    setError('');
    setCurrentStep(2);
  };

  const handlePetComplete = async (data: PetInfoData) => {
    if (!ownerData && !isAuthenticated) {
      setError('Your account details are missing. Return to the first step and try again.');
      return;
    }

    setPetData(data);
    setIsLoading(true);
    setError('');

    try {
      if (!useAuthStore.getState().isAuthenticated) {
        if (!ownerData?.password) {
          throw new Error(
            'Your account password is missing. Return to the first step and try again.'
          );
        }
        const registrationRequest = {
          handle: ownerData.handle,
          email: ownerData.email,
          password: ownerData.password,
          bio: ownerData.bio || undefined,
          registrationKey: getRegistrationKey(),
        };
        await authApi.register(registrationRequest);
        clearRegistrationReplayKey();

        // The credential has done its job. Do not retain plaintext password for the rest of onboarding.
        setOwnerData((current) => (current ? { ...current, password: '' } : current));
      }

      let pair = durablePair;
      if (pair) {
        await petsApi.updatePet(pair.petId, {
          name: data.name,
          species: data.species,
          breed: data.breed,
          birthdate: data.birthdate,
        });
        pair = { ...pair, petName: data.name };
        rememberPair(pair);
      } else {
        const pet = await petsApi.createPet({
          name: data.name,
          species: data.species,
          breed: data.breed || undefined,
          birthdate: data.birthdate,
          creationKey: getCreationKey(),
        });
        const householdId = pet.householdMemberships[0]?.householdId;
        if (!householdId) {
          throw new Error('Woof created the pet but could not resolve its household pair.');
        }

        pair = { petId: pet.id, householdId, petName: pet.name };
        rememberPair(pair);
      }

      await attachPhoto(pair, data.photoFile);
      await refreshAuthProfile();
      setCurrentStep(3);
    } catch (err: unknown) {
      console.error('Durable onboarding setup failed', err);
      const message = apiErrorMessage(
        err,
        'We could not finish creating the pair. Your completed details are preserved, and exact retries reuse transaction keys instead of creating duplicate accounts or pets.'
      );
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const finishFirstAdventure = async (selections: FirstAdventureSelections, skipAll = false) => {
    if (!durablePair) {
      setError('Your pet pair is missing. Return to the pet step so Woof can restore it safely.');
      return;
    }

    setIsLoading(true);
    setError('');

    const responses = buildFirstAdventureResponses(durablePair.petId, selections, skipAll);
    const results = await Promise.allSettled(
      responses.map((response) =>
        adaptiveProfileApi.recordQuestionResponse(
          durablePair.householdId,
          durablePair.petId,
          response
        )
      )
    );

    const failedWrites = results.filter((result) => result.status === 'rejected');
    if (failedWrites.length > 0) {
      console.warn('Some First Adventure profile signals were not saved', failedWrites);
      toast.warning(
        'Your pair is ready. A few optional preferences did not save, so Woof will simply learn them later.'
      );
    } else if (skipAll) {
      toast.success(`You and ${durablePair.petName} are ready. We can learn as we go. 🐾`);
    } else {
      toast.success(`You and ${durablePair.petName} are ready for something real. 🐾`);
    }

    await refreshAuthProfile();
    clearRecoveryHints();
    router.replace('/');
  };

  const handleBack = () => {
    if (isLoading) return;

    if (currentStep === 3) {
      setCurrentStep(2);
      return;
    }

    if (currentStep === 2 && useAuthStore.getState().isAuthenticated) {
      router.push('/');
      return;
    }

    if (currentStep > 1) {
      setCurrentStep((step) => step - 1);
    } else {
      router.push('/login');
    }
  };

  const backLabel =
    currentStep === 3
      ? 'Review pet details'
      : currentStep === 2 && isAuthenticated
        ? 'Leave setup'
        : currentStep > 1
          ? 'Go to previous onboarding step'
          : 'Return to sign in';

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-lg items-center gap-4 px-4 py-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleBack}
            disabled={isLoading}
            className="shrink-0 rounded-xl"
            aria-label={backLabel}
          >
            <ChevronLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
          <div className="flex-1">
            <div className="mb-2 flex items-center justify-between gap-4">
              <span className="text-sm font-semibold">
                {currentStep === 3 ? 'First Adventure' : `Step ${currentStep} of ${totalSteps}`}
              </span>
              <span className="text-xs text-muted-foreground">{Math.round(progress)}%</span>
            </div>
            <Progress
              value={progress}
              className="h-2"
              aria-label={`Onboarding ${Math.round(progress)}% complete`}
            />
          </div>
          <span
            className="brand-mark flex h-9 w-9 shrink-0 items-center justify-center rounded-xl"
            title="Woof"
          >
            <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-lg px-4 py-8">
        {currentStep === 1 && (
          <OwnerInfoStep onComplete={handleOwnerComplete} initialData={ownerData} />
        )}
        {currentStep === 2 && (
          <PetInfoStep
            onComplete={(data) => void handlePetComplete(data)}
            initialData={petData}
            isLoading={isLoading}
          />
        )}
        {currentStep === 3 && durablePair && (
          <FirstAdventureStep
            petName={durablePair.petName}
            isLoading={isLoading}
            onComplete={(selections) => void finishFirstAdventure(selections)}
            onSkipAll={() => void finishFirstAdventure(emptyFirstAdventureSelections(), true)}
          />
        )}

        {error && currentStep >= 2 && (
          <div
            role="alert"
            aria-live="polite"
            className="mt-5 rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm leading-relaxed text-destructive"
          >
            {error}
          </div>
        )}
      </main>
    </div>
  );
}
