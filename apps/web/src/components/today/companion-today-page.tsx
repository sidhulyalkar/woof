'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { BookOpenCheck, Gamepad2, HeartHandshake, Home, PawPrint, Users } from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { CaregiverAccessPanel } from '@/components/caregiver/caregiver-access-panel';
import { Button } from '@/components/ui/button';
import { companionApi, type CompanionState } from '@/lib/api/companion';
import { socialAdventureApi } from '@/lib/api/social-adventure';

const modeCopy = {
  ANIMAL_ALLY: {
    eyebrow: 'Animal Ally',
    title: 'Learn useful dog-human skills before you need a pet profile.',
    body: 'Practice observation and handling judgment, join the community, and build practical readiness without creating fictional dog data.',
    icon: HeartHandshake,
  },
  FOSTER_CAREGIVER: {
    eyebrow: 'Foster Caregiver',
    title: 'Prepare for temporary care without pretending every dog is the same.',
    body: 'Build human skill and practical readiness first. Individual foster-dog assumptions should begin only after an authorized relationship exists.',
    icon: Home,
  },
} as const;

export default function CompanionTodayPage({ state }: { state: CompanionState }) {
  const queryClient = useQueryClient();
  const social = useQuery({
    queryKey: ['social-adventure', 'me'],
    queryFn: socialAdventureApi.getMine,
    retry: false,
  });
  const modeMutation = useMutation({
    mutationFn: companionApi.updateMode,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['companion'] });
    },
  });

  const mode = state.mode === 'FOSTER_CAREGIVER' ? 'FOSTER_CAREGIVER' : 'ANIMAL_ALLY';
  const copy = modeCopy[mode];
  const Icon = copy.icon;
  const completedSkills = social.data?.components.humanSkill.completedChallenges.length ?? 0;

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
            <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Companion mode
            </p>
            <h1 className="text-lg font-bold tracking-tight">Today</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        {/* A real pending/active caregiver grant outranks generic learning content.
            The panel renders nothing when no authority exists, preserving the normal companion layout. */}
        <CaregiverAccessPanel />

        <section className="mt-6 rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.1] via-card/95 to-secondary/[0.06] p-6">
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Icon className="h-5 w-5" aria-hidden="true" />
          </span>
          <p className="eyebrow mt-4">{copy.eyebrow}</p>
          <h2 className="mt-1 text-3xl font-bold tracking-tight text-balance">{copy.title}</h2>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{copy.body}</p>
        </section>

        <section className="mt-6 grid gap-3 sm:grid-cols-2">
          <Link
            href="/arcade"
            className="surface-soft rounded-2xl p-5 transition hover:ring-2 hover:ring-primary/20"
          >
            <Gamepad2 className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-bold">Skillcraft Arcade</h2>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              Practice marker timing, making setups easier, catching useful behavior, and positive
              association timing.
            </p>
            <p className="mt-3 text-xs font-semibold text-primary">
              {completedSkills}/4 games explored this week →
            </p>
          </Link>

          <Link
            href="/community"
            className="surface-soft rounded-2xl p-5 transition hover:ring-2 hover:ring-primary/20"
          >
            <Users className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-bold">Community</h2>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              Join Packs and learn from shared good reads without competing on pet performance.
            </p>
            <p className="mt-3 text-xs font-semibold text-primary">Open Community →</p>
          </Link>

          <Link
            href="/companion/readiness"
            className="surface-soft rounded-2xl p-5 transition hover:ring-2 hover:ring-primary/20 sm:col-span-2"
          >
            <BookOpenCheck className="h-5 w-5 text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-bold">Private readiness reflection</h2>
            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
              Think through housing, time, finances, household alignment, support, and care
              planning. No score and no automatic sharing.
            </p>
            <p className="mt-3 text-xs font-semibold text-primary">Reflect on readiness →</p>
          </Link>
        </section>

        <section className="mt-6 rounded-3xl border border-border/70 bg-card/60 p-5">
          <p className="eyebrow">Real-world opportunities</p>
          <h2 className="mt-1 text-lg font-bold">Partner authority before placement prompts.</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Woof does not fabricate shelter inventory or scrape a dog into your profile. Foster,
            volunteer, and adoption opportunities will appear only from authorized partner sources
            with their own eligibility and placement authority.
          </p>
        </section>

        <section className="mt-6 rounded-3xl border border-border/70 bg-card/50 p-5">
          <h2 className="font-bold">Your role can change.</h2>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            If you now live with a dog, switch to Pet Guardian. Woof will still require a real pet
            or authorized household relationship before opening pet-specific surfaces.
          </p>
          <Button
            variant="outline"
            className="mt-4 bg-transparent"
            disabled={modeMutation.isPending}
            onClick={() => modeMutation.mutate('PET_GUARDIAN')}
          >
            I have a dog now
          </Button>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
