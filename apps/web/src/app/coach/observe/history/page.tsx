'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Brain, Camera, Film, Loader2, Trash2 } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { behaviorVisionApi } from '@/lib/api/behavior-vision';
import { useSessionStore } from '@/store/session';

export default function BehaviorHistoryPage() {
  const user = useSessionStore((state) => state.user);
  const pets = user?.pets ?? [];
  const [petId, setPetId] = useState(pets[0]?.id ?? '');
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!petId && pets[0]) setPetId(pets[0].id);
  }, [petId, pets]);

  const timelineQuery = useQuery({
    queryKey: ['behavior-timeline', petId],
    queryFn: () => behaviorVisionApi.timeline(petId, 50),
    enabled: Boolean(petId),
  });
  const profileQuery = useQuery({
    queryKey: ['behavior-profile', petId],
    queryFn: () => behaviorVisionApi.profile(petId),
    enabled: Boolean(petId),
  });

  const deleteMutation = useMutation({
    mutationFn: (observationId: string) => behaviorVisionApi.deleteObservation(observationId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['behavior-timeline', petId] });
      void queryClient.invalidateQueries({ queryKey: ['behavior-profile', petId] });
    },
  });

  const timeline = timelineQuery.data ?? [];
  const profile = profileQuery.data;

  return (
    <div className="min-h-screen bg-background pb-28">
      <header className="sticky top-0 z-20 border-b border-border/50 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-xl items-center gap-3 px-4 py-4">
          <Button asChild variant="ghost" size="icon">
            <Link href="/coach/observe" aria-label="Back to behavior capture">
              <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">Individual model</p>
            <h1 className="truncate text-lg font-bold">Behavior history</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl space-y-5 px-4 py-5">
        {pets.length > 1 && (
          <label className="block text-sm font-semibold">
            Pet
            <select
              value={petId}
              onChange={(event) => setPetId(event.target.value)}
              className="mt-2 h-11 w-full rounded-xl border border-border bg-card px-3 text-sm font-normal"
            >
              {pets.map((pet) => (
                <option key={pet.id} value={pet.id}>
                  {pet.name}
                </option>
              ))}
            </select>
          </label>
        )}

        {profile && (
          <section className="rounded-3xl border border-primary/15 bg-primary/[0.055] p-5">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                <Brain className="h-5 w-5" aria-hidden="true" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="font-semibold">What Woof has learned</h2>
                  <span className="text-xs font-semibold text-primary">
                    {Math.round(profile.personalizationConfidence * 100)}%
                  </span>
                </div>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {profile.sampleCount} usable observations across {profile.contextsSeen.length} contexts.
                  Confidence reflects evidence depth and breadth, not a personality score.
                </p>
              </div>
            </div>

            {profile.interventionEffects.length > 0 && (
              <div className="mt-4 space-y-2">
                {profile.interventionEffects.slice(0, 4).map((effect) => (
                  <div key={effect.action} className="rounded-xl border border-border/50 bg-background/60 p-3">
                    <div className="flex items-center justify-between gap-3 text-xs">
                      <span className="font-semibold">{effect.action}</span>
                      <span className="text-muted-foreground">{effect.pairedSessions} paired sessions</span>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Evidence confidence {Math.round(effect.confidence * 100)}%
                    </p>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        <section>
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="font-semibold">Observation timeline</h2>
              <p className="mt-1 text-xs text-muted-foreground">Derived observations only. Raw media is not stored here.</p>
            </div>
          </div>

          {timelineQuery.isLoading && (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" aria-label="Loading behavior history" />
            </div>
          )}

          {!timelineQuery.isLoading && timeline.length === 0 && (
            <div className="mt-4 rounded-3xl border border-dashed border-border p-8 text-center">
              <p className="font-semibold">No observations yet</p>
              <p className="mt-1 text-sm text-muted-foreground">
                A few short, comparable clips are more useful than one long video.
              </p>
              <Button asChild className="mt-4">
                <Link href="/coach/observe">Capture an observation</Link>
              </Button>
            </div>
          )}

          <div className="mt-4 space-y-3">
            {timeline.map((entry) => (
              <article key={entry.id} className="rounded-2xl border border-border/60 bg-card/65 p-4">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-muted text-muted-foreground">
                    {entry.mediaType === 'video' ? (
                      <Film className="h-4 w-4" aria-hidden="true" />
                    ) : (
                      <Camera className="h-4 w-4" aria-hidden="true" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold">
                          {entry.context.context} · {entry.context.phase}
                        </p>
                        <p className="mt-0.5 text-xs text-muted-foreground">
                          {new Date(entry.createdAt).toLocaleString()}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => deleteMutation.mutate(entry.id)}
                        disabled={deleteMutation.isPending}
                        className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                        aria-label="Delete behavior observation"
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                    <p className="mt-3 text-sm leading-relaxed">{entry.analysis.observableSummary}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {entry.analysis.dimensions
                        .filter((dimension) => dimension.confidence >= 0.5)
                        .slice(0, 4)
                        .map((dimension) => (
                          <span
                            key={dimension.dimension}
                            className="rounded-full border border-border/60 bg-background px-2.5 py-1 text-[11px] font-medium text-muted-foreground"
                          >
                            {dimension.dimension} {Math.round(dimension.value * 100)}%
                          </span>
                        ))}
                    </div>
                    {entry.ownerFeedback && (
                      <p className="mt-3 text-xs font-medium text-muted-foreground">
                        Owner feedback: {entry.ownerFeedback.accurate ? 'matched what I saw' : 'needs correction'}
                      </p>
                    )}
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
