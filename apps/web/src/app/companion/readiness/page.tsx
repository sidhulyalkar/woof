'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { BookOpenCheck, Loader2, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { companionApi, type ReadinessDimension, type ReadinessStatus } from '@/lib/api/companion';

const dimensions: Array<{
  key: ReadinessDimension;
  label: string;
  description: string;
}> = [
  {
    key: 'housing',
    label: 'Housing',
    description:
      'Would the living situation realistically support the animal and applicable rules?',
  },
  {
    key: 'householdAlignment',
    label: 'Household alignment',
    description: 'Are the people sharing the home aligned on responsibilities and boundaries?',
  },
  {
    key: 'timeCapacity',
    label: 'Time capacity',
    description:
      'Is there room for daily care, decompression, training, transport, and interruptions?',
  },
  {
    key: 'financialPlan',
    label: 'Financial plan',
    description: 'Is there a realistic plan for routine costs and unexpected care?',
  },
  {
    key: 'supportPlan',
    label: 'Support plan',
    description: 'Who can help with care, transport, travel, emergencies, or difficult weeks?',
  },
  {
    key: 'carePlan',
    label: 'Care plan',
    description:
      'Are veterinary care, supplies, management, and transition needs understood enough to discuss?',
  },
];

const statuses: Array<{ value: ReadinessStatus; label: string }> = [
  { value: 'NOT_SURE', label: 'Not sure yet' },
  { value: 'WORKING_ON_IT', label: 'Working on it' },
  { value: 'READY_TO_DISCUSS', label: 'Ready to discuss' },
];

export default function CompanionReadinessPage() {
  const queryClient = useQueryClient();
  const readiness = useQuery({
    queryKey: ['companion', 'readiness'],
    queryFn: companionApi.readiness,
    retry: false,
  });
  const update = useMutation({
    mutationFn: ({ key, value }: { key: ReadinessDimension; value: ReadinessStatus }) =>
      companionApi.updateReadiness({ [key]: value }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['companion', 'readiness'] });
    },
  });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <BookOpenCheck className="h-5 w-5 text-primary" aria-hidden="true" />
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Companion mode
            </p>
            <h1 className="text-lg font-bold tracking-tight">Readiness reflection</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/20 bg-primary/[0.05] p-5">
          <div className="flex items-start gap-3">
            <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-bold">A checklist, not a verdict.</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                These answers are private self-reflection. Woof does not combine them into an
                adoption or foster score, publish them, or replace a shelter, rescue, landlord,
                veterinarian, trainer, or financial decision-maker.
              </p>
            </div>
          </div>
          <Button asChild variant="outline" className="mt-4 bg-transparent">
            <Link href="/">← Back to Today</Link>
          </Button>
        </section>

        {readiness.isLoading ? (
          <div className="flex min-h-48 items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
          </div>
        ) : readiness.isError || !readiness.data ? (
          <section className="surface-soft mt-5 rounded-2xl p-5 text-center">
            <p className="font-semibold">Your reflection is unavailable right now.</p>
            <Button className="mt-4" onClick={() => void readiness.refetch()}>
              Try again
            </Button>
          </section>
        ) : (
          <div className="mt-5 space-y-3">
            {dimensions.map((dimension) => {
              const current = readiness.data.dimensions[dimension.key];
              return (
                <section key={dimension.key} className="surface-soft rounded-2xl p-5">
                  <h2 className="font-bold">{dimension.label}</h2>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    {dimension.description}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {statuses.map((status) => (
                      <Button
                        key={status.value}
                        type="button"
                        size="sm"
                        variant={current === status.value ? 'default' : 'outline'}
                        className={current === status.value ? undefined : 'bg-transparent'}
                        disabled={update.isPending}
                        onClick={() => update.mutate({ key: dimension.key, value: status.value })}
                      >
                        {status.label}
                      </Button>
                    ))}
                  </div>
                </section>
              );
            })}
          </div>
        )}

        <p className="mt-5 text-xs leading-relaxed text-muted-foreground">
          A “Ready to discuss” selection means only that you feel prepared to talk about that
          dimension. Placement and foster decisions stay with the organizations and people
          responsible for them.
        </p>
      </main>

      <BottomNav />
    </div>
  );
}
