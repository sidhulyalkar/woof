'use client';

import { isAxiosError } from 'axios';
import { ChevronLeft, Loader2, PawPrint } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useRef, useState } from 'react';
import { CompanionModeChooser } from '@/components/companion/companion-mode-chooser';
import { OwnerInfoStep, type OwnerInfoData } from '@/components/onboarding/owner-info-step';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { authApi } from '@/lib/api';
import { companionApi, type CompanionMode } from '@/lib/api/companion';
import { useAuthStore } from '@/lib/stores/auth-store';

type ApiErrorBody = { message?: string | string[] };
const REGISTRATION_KEY_STORAGE = 'woof:companion-onramp:registration-key';

function apiErrorMessage(error: unknown, fallback: string) {
  const responseMessage = isAxiosError<ApiErrorBody>(error)
    ? error.response?.data?.message
    : undefined;
  if (Array.isArray(responseMessage)) return responseMessage.join(' ');
  return responseMessage || fallback;
}

export default function CompanionOnboardingPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [ownerData, setOwnerData] = useState<OwnerInfoData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const registrationKeyRef = useRef<string | null>(null);

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

  const clearRegistrationKey = () => {
    window.sessionStorage.removeItem(REGISTRATION_KEY_STORAGE);
    registrationKeyRef.current = null;
  };

  const selectMode = async (mode: CompanionMode) => {
    if (!ownerData && !useAuthStore.getState().isAuthenticated) return;
    setIsLoading(true);
    setError('');

    try {
      if (!useAuthStore.getState().isAuthenticated) {
        if (!ownerData?.password) throw new Error('Your account password is missing.');
        const registrationRequest = {
          handle: ownerData.handle,
          email: ownerData.email,
          password: ownerData.password,
          bio: ownerData.bio || undefined,
          registrationKey: getRegistrationKey(),
        };
        await authApi.register(registrationRequest);
        clearRegistrationKey();
        setOwnerData((current) => (current ? { ...current, password: '' } : current));
      }

      await companionApi.updateMode(mode);
      router.replace(mode === 'PET_GUARDIAN' ? '/onboarding' : '/');
    } catch (caught) {
      setError(
        apiErrorMessage(
          caught,
          'We could not finish your account mode. Your pet and relationship state were not changed.'
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const progress = step === 1 ? 50 : 100;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-lg items-center gap-4 px-4 py-4">
          <Button
            variant="ghost"
            size="icon"
            disabled={isLoading}
            onClick={() => (step === 2 ? setStep(1) : router.push('/login'))}
            aria-label={step === 2 ? 'Back to account details' : 'Return to sign in'}
          >
            <ChevronLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
          <div className="flex-1">
            <div className="mb-2 flex items-center justify-between gap-4">
              <span className="text-sm font-semibold">Step {step} of 2</span>
              <span className="text-xs text-muted-foreground">{progress}%</span>
            </div>
            <Progress
              value={progress}
              className="h-2"
              aria-label={`Onboarding ${progress}% complete`}
            />
          </div>
          <span className="brand-mark flex h-9 w-9 shrink-0 items-center justify-center rounded-xl">
            <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-lg px-4 py-8">
        {step === 1 ? (
          <OwnerInfoStep
            initialData={ownerData}
            onComplete={(data) => {
              setOwnerData(data);
              setError('');
              setStep(2);
            }}
          />
        ) : (
          <section>
            <p className="eyebrow">Choose your starting role</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              You do not need a pet to start learning.
            </h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              This choice controls the experience you see. It never creates pet access. Pet-specific
              dogOS opens only after a real owned or authorized household relationship exists.
            </p>
            <div className="mt-6">
              <CompanionModeChooser
                disabled={isLoading}
                onSelect={(mode) => void selectMode(mode)}
              />
            </div>
            {isLoading && (
              <p
                className="mt-4 flex items-center gap-2 text-sm text-muted-foreground"
                role="status"
              >
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                Saving your starting role…
              </p>
            )}
          </section>
        )}

        {error && (
          <div
            role="alert"
            className="mt-5 rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive"
          >
            {error}
          </div>
        )}
      </main>
    </div>
  );
}
