'use client';

import { useQuery } from '@tanstack/react-query';
import {
  CalendarHeart,
  Compass,
  HeartHandshake,
  Loader2,
  PawPrint,
  Sparkles,
  Users,
} from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { packApi } from '@/lib/api/pack';

export default function PackPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['pack', 'challenges'],
    queryFn: packApi.challenges,
    retry: false,
  });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Users className="h-5 w-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Together, without a podium
            </p>
            <h1 className="text-lg font-bold tracking-tight">Pack</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.1] via-card/90 to-secondary/[0.07] p-5">
          <p className="eyebrow">Cooperative play</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight">
            Everybody can move the pack forward.
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Pack challenges count appropriate shared actions, not raw distance. A senior dog, a
            cautious dog, and an adolescent trail rocket can all contribute without pretending their
            ideal weeks should look alike.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-2">
            <Button asChild>
              <Link href="/discover">
                <Compass className="mr-2 h-4 w-4" aria-hidden="true" />
                Find a good match
              </Link>
            </Button>
            <Button variant="outline" asChild className="bg-transparent">
              <Link href="/events">
                <CalendarHeart className="mr-2 h-4 w-4" aria-hidden="true" />
                Plan together
              </Link>
            </Button>
          </div>
        </section>

        <section className="mt-6">
          <div>
            <p className="eyebrow">This week</p>
            <h2 className="mt-1 text-xl font-bold tracking-tight">Community quests</h2>
          </div>

          {isLoading ? (
            <div className="flex min-h-48 items-center justify-center" role="status">
              <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
            </div>
          ) : error || !data ? (
            <div className="surface-soft mt-3 rounded-2xl p-5 text-center">
              <PawPrint className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
              <p className="mt-2 font-semibold">Pack totals are unavailable.</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Your individual Adventure system still works normally.
              </p>
            </div>
          ) : (
            <div className="mt-3 space-y-3">
              {data.challenges.map((challenge) => (
                <article key={challenge.id} className="surface-soft rounded-3xl p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-4 w-4 text-primary" aria-hidden="true" />
                        <h3 className="font-bold">{challenge.title}</h3>
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                        {challenge.description}
                      </p>
                    </div>
                    {challenge.completed && (
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                        Done
                      </span>
                    )}
                  </div>

                  <Progress className="mt-4 h-2" value={challenge.progress * 100} />
                  <div className="mt-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                    <span>
                      {challenge.total} / {challenge.target} {challenge.unit}
                    </span>
                    <span>{challenge.contributors} contributors</span>
                  </div>
                  <div className="mt-3 rounded-2xl bg-primary/[0.06] px-3 py-2 text-sm">
                    <span className="font-semibold text-primary">Your contribution:</span>{' '}
                    {challenge.myContribution} meaningful{' '}
                    {challenge.myContribution === 1 ? 'action' : 'actions'}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="mt-6 rounded-3xl border border-border/60 bg-card/55 p-5">
          <div className="flex items-start gap-3">
            <HeartHandshake className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-bold">Social fit still comes first</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Pack does not reward forcing interactions. Parallel walks, extra distance, ending
                early, or skipping a crowded event can all be better choices for a particular dog.
              </p>
              <Button variant="ghost" size="sm" asChild className="mt-2 px-0 text-primary">
                <Link href="/discover">Explore compatibility-first discovery →</Link>
              </Button>
            </div>
          </div>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
