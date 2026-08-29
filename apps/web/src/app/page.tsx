'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, PawPrint } from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { CompanionModeChooser } from '@/components/companion/companion-mode-chooser';
import CompanionTodayPage from '@/components/today/companion-today-page';
import GuardianTodayPage from '@/components/today/guardian-today-page';
import { Button } from '@/components/ui/button';
import { companionApi, type CompanionMode } from '@/lib/api/companion';

export default function HomePage() {
  const queryClient = useQueryClient();
  const state = useQuery({
    queryKey: ['companion', 'state'],
    queryFn: companionApi.state,
    retry: false,
  });

  const modeMutation = useMutation({
    mutationFn: (mode: CompanionMode) => companionApi.updateMode(mode),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['companion'] });
    },
  });

  if (state.isLoading) {
    return (
      <main
        id="main-content"
        className="flex min-h-screen items-center justify-center"
        role="status"
      >
        <div className="text-center">
          <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
          <p className="mt-3 text-sm text-muted-foreground">Opening the right Woof for you…</p>
        </div>
      </main>
    );
  }

  if (state.isError || !state.data) {
    return (
      <div className="min-h-screen pb-24">
        <main id="main-content" className="mx-auto max-w-xl px-4 py-16">
          <section className="surface-soft rounded-3xl p-6 text-center">
            <PawPrint className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h1 className="mt-3 text-xl font-bold">We could not resolve your Woof mode</h1>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Pet-only surfaces stay closed when account mode cannot be verified. Retry when the
              connection is available.
            </p>
            <Button className="mt-5" onClick={() => void state.refetch()}>
              Try again
            </Button>
          </section>
        </main>
        <BottomNav />
      </div>
    );
  }

  if (state.data.landing === 'PET_TODAY') {
    return <GuardianTodayPage />;
  }

  if (state.data.landing === 'COMPANION_TODAY') {
    return <CompanionTodayPage state={state.data} />;
  }

  if (state.data.landing === 'NEEDS_PET_SETUP') {
    return (
      <div className="min-h-screen pb-24">
        <main id="main-content" className="mx-auto max-w-xl px-4 py-10">
          <section className="rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.1] via-card/95 to-secondary/[0.06] p-6">
            <p className="eyebrow">Pet Guardian</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Add the dog you actually care for.
            </h1>
            <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
              Choosing Pet Guardian never creates pet authority by itself. Add a real pet, or join a
              household through an authorized relationship, before pet-specific Today opens.
            </p>
            <Button asChild className="mt-5">
              <Link href="/onboarding">Add a dog</Link>
            </Button>
          </section>

          <section className="mt-6">
            <p className="eyebrow">Different role?</p>
            <h2 className="mt-1 text-xl font-bold">Use Woof without inventing a pet.</h2>
            <div className="mt-3">
              <CompanionModeChooser
                compact
                disabled={modeMutation.isPending}
                onSelect={(mode) => modeMutation.mutate(mode)}
              />
            </div>
          </section>
        </main>
        <BottomNav />
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-24">
      <main id="main-content" className="mx-auto max-w-xl px-4 py-10">
        <section>
          <p className="eyebrow">Welcome to Woof</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">How do you want to use dogOS?</h1>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            Pick the role that fits today. This controls the experience you see, not which pets you
            are authorized to access. You can change it later.
          </p>
          <div className="mt-6">
            <CompanionModeChooser
              disabled={modeMutation.isPending}
              onSelect={(mode) => modeMutation.mutate(mode)}
            />
          </div>
          {modeMutation.isError && (
            <p className="mt-4 text-sm text-destructive" role="alert">
              Your mode could not be saved. No pet or relationship state was changed.
            </p>
          )}
        </section>
      </main>
      <BottomNav />
    </div>
  );
}
