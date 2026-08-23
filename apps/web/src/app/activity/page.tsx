'use client';

import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  CheckCircle2,
  Clock3,
  History,
  Loader2,
  PawPrint,
  Plus,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { PetSwitcher } from '@/components/pets/pet-switcher';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { getActivePetId, setActivePetId } from '@/lib/active-pet';
import { activitiesApi, type CanonicalActivity } from '@/lib/api/activities';
import { petsApi } from '@/lib/api/pets';

const QUICK_TYPES = ['WALK', 'PLAY', 'TRAINING', 'RECOVERY'] as const;
const QUICK_DURATIONS = [15, 30, 45, 60] as const;

function friendlyType(type: string) {
  return type
    .toLowerCase()
    .split('_')
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(' ');
}

function activityDuration(activity: CanonicalActivity) {
  if (!activity.endedAt) return null;
  const start = new Date(activity.startedAt).getTime();
  const end = new Date(activity.endedAt).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  return Math.round((end - start) / 60_000);
}

function activityDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Saved activity';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

export default function ActivityPage() {
  const queryClient = useQueryClient();
  const [selectedPetId, setSelectedPetId] = useState<string | null>(null);
  const [quickType, setQuickType] = useState<(typeof QUICK_TYPES)[number]>('WALK');
  const [quickDuration, setQuickDuration] = useState<(typeof QUICK_DURATIONS)[number]>(30);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);

  const pets = useQuery({
    queryKey: ['pets', 'mine'],
    queryFn: () => petsApi.getMine(),
    staleTime: 60_000,
    retry: false,
  });

  useEffect(() => {
    const ownedPets = pets.data?.pets ?? [];
    if (ownedPets.length === 0) {
      setSelectedPetId(null);
      return;
    }

    const remembered = getActivePetId();
    const next = ownedPets.some((pet) => pet.id === remembered) ? remembered : ownedPets[0]?.id;
    if (!next || next === selectedPetId) return;

    setActivePetId(next);
    setSelectedPetId(next);
  }, [pets.data, selectedPetId]);

  const activities = useInfiniteQuery({
    queryKey: ['activities', selectedPetId],
    enabled: Boolean(selectedPetId),
    initialPageParam: 0,
    queryFn: ({ pageParam }) =>
      activitiesApi.getMine({ petId: selectedPetId ?? undefined, skip: pageParam, take: 20 }),
    getNextPageParam: (lastPage) => {
      const next = lastPage.skip + lastPage.activities.length;
      return next < lastPage.total ? next : undefined;
    },
    retry: false,
  });

  const activityList = useMemo(
    () => activities.data?.pages.flatMap((page) => page.activities) ?? [],
    [activities.data]
  );
  const totalActivities = activities.data?.pages[0]?.total ?? 0;
  const selectedPet = pets.data?.pets.find((pet) => pet.id === selectedPetId) ?? null;

  const quickLog = useMutation({
    mutationFn: async () => {
      if (!selectedPetId) throw new Error('Choose a dog first');
      const endedAt = new Date();
      const startedAt = new Date(endedAt.getTime() - quickDuration * 60_000);
      return activitiesApi.create({
        petIds: [selectedPetId],
        type: quickType,
        startedAt: startedAt.toISOString(),
        endedAt: endedAt.toISOString(),
        jointMetrics: {
          source: 'MANUAL_QUICK_LOG',
          enteredDurationMinutes: quickDuration,
        },
      });
    },
    onSuccess: async () => {
      setSavedMessage(
        `${friendlyType(quickType)} saved for ${selectedPet?.name ?? 'your dog'}. Today, Adventure, and Story can now use it.`
      );
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['activities', selectedPetId] }),
        queryClient.invalidateQueries({ queryKey: ['adventure', 'me'] }),
        queryClient.invalidateQueries({ queryKey: ['concierge', 'today'] }),
        queryClient.invalidateQueries({ queryKey: ['story'] }),
      ]);
    },
  });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <div>
            <p className="eyebrow">Canonical history</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Activity</h1>
          </div>
          {selectedPet && (
            <span className="rounded-full border border-border/70 bg-card/70 px-3 py-1.5 text-xs font-semibold">
              {selectedPet.name}
            </span>
          )}
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl space-y-5 px-4 py-5">
        {pets.isLoading ? (
          <Card className="surface-soft flex items-center gap-3 rounded-3xl p-5" role="status">
            <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Loading your dogs…</span>
          </Card>
        ) : pets.isError ? (
          <Card className="surface-soft rounded-3xl p-5 text-center">
            <PawPrint className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-semibold">We couldn&apos;t load your dogs</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Activity history stays hidden rather than guessing which pet it belongs to.
            </p>
            <Button className="mt-4" variant="outline" onClick={() => pets.refetch()}>
              <RefreshCw className="mr-2 h-4 w-4" aria-hidden="true" />
              Try again
            </Button>
          </Card>
        ) : (pets.data?.pets.length ?? 0) === 0 ? (
          <Card className="surface-soft rounded-3xl p-6 text-center">
            <PawPrint className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h2 className="mt-3 text-lg font-semibold">Add your first dog to begin</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Woof keeps activity attached to a real pet profile. Nothing is logged into an
              anonymous bucket.
            </p>
            <Button className="mt-5" asChild>
              <Link href="/pets/new">Add your dog</Link>
            </Button>
          </Card>
        ) : selectedPet ? (
          <>
            <Card className="rounded-3xl border-primary/15 bg-gradient-to-br from-primary/[0.07] via-card/95 to-secondary/[0.04] p-5">
              <div className="flex items-start gap-3">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <Plus className="h-5 w-5" aria-hidden="true" />
                </span>
                <div>
                  <p className="eyebrow">Quick log</p>
                  <h2 className="mt-1 text-lg font-bold">Save what just happened</h2>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    A few taps create a real completed Activity. No fake routes, no inferred
                    distance, and no background claim about where you went.
                  </p>
                </div>
              </div>

              <PetSwitcher
                currentPetId={selectedPet.id}
                label="Logging for"
                onChange={(petId) => {
                  setSavedMessage(null);
                  setSelectedPetId(petId);
                }}
              />

              <div className="mt-4">
                <p className="text-xs font-semibold text-muted-foreground">What did you do?</p>
                <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label="Activity type">
                  {QUICK_TYPES.map((type) => (
                    <button
                      key={type}
                      type="button"
                      aria-pressed={quickType === type}
                      onClick={() => setQuickType(type)}
                      className={`rounded-full border px-3 py-2 text-sm font-semibold transition-colors ${
                        quickType === type
                          ? 'border-primary/30 bg-primary/10 text-primary'
                          : 'border-border/70 bg-background/55 text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {friendlyType(type)}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-4">
                <p className="text-xs font-semibold text-muted-foreground">About how long?</p>
                <div className="mt-2 grid grid-cols-4 gap-2" role="group" aria-label="Activity duration">
                  {QUICK_DURATIONS.map((minutes) => (
                    <button
                      key={minutes}
                      type="button"
                      aria-pressed={quickDuration === minutes}
                      onClick={() => setQuickDuration(minutes)}
                      className={`rounded-2xl border px-2 py-2 text-sm font-semibold transition-colors ${
                        quickDuration === minutes
                          ? 'border-primary/30 bg-primary/10 text-primary'
                          : 'border-border/70 bg-background/55 text-muted-foreground hover:text-foreground'
                      }`}
                    >
                      {minutes}m
                    </button>
                  ))}
                </div>
              </div>

              <Button
                className="mt-4 w-full"
                size="lg"
                disabled={quickLog.isPending}
                onClick={() => quickLog.mutate()}
              >
                {quickLog.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <CheckCircle2 className="mr-2 h-4 w-4" aria-hidden="true" />
                )}
                Save {quickDuration} min {friendlyType(quickType).toLowerCase()}
              </Button>

              {quickLog.isError && (
                <p className="mt-3 text-sm text-destructive" role="alert">
                  That activity could not be saved. Nothing was added to history.
                </p>
              )}
              {savedMessage && (
                <div className="mt-3 flex items-start gap-2 rounded-2xl bg-primary/10 p-3 text-sm text-primary">
                  <Sparkles className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
                  <span>{savedMessage}</span>
                </div>
              )}
            </Card>

            <section aria-labelledby="activity-history-heading">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">What actually happened</p>
                  <h2 id="activity-history-heading" className="mt-1 text-lg font-bold">
                    {selectedPet.name}&apos;s history
                  </h2>
                </div>
                <span className="text-xs text-muted-foreground">
                  {totalActivities} record{totalActivities === 1 ? '' : 's'}
                </span>
              </div>

              {activities.isLoading ? (
                <Card className="surface-soft mt-3 flex items-center gap-3 rounded-2xl p-4" role="status">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
                  <span className="text-sm text-muted-foreground">Loading canonical history…</span>
                </Card>
              ) : activities.isError ? (
                <Card className="surface-soft mt-3 rounded-2xl p-5 text-center">
                  <History className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
                  <p className="mt-3 font-semibold">History is temporarily unavailable</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Woof will not substitute demo data while the real history cannot be read.
                  </p>
                  <Button className="mt-4" variant="outline" onClick={() => activities.refetch()}>
                    Try again
                  </Button>
                </Card>
              ) : activityList.length === 0 ? (
                <Card className="surface-soft mt-3 rounded-2xl p-5 text-center">
                  <History className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
                  <p className="mt-3 font-semibold">No activity yet</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Your first real walk, play session, training session, or recovery block will
                    appear here after it is saved.
                  </p>
                </Card>
              ) : (
                <div className="mt-3 space-y-2">
                  {activityList.map((activity) => {
                    const minutes = activityDuration(activity);
                    const participantNames = activity.petParticipants
                      .map((participant) => participant.pet.name)
                      .filter(Boolean);
                    return (
                      <Card key={activity.id} className="surface-soft rounded-2xl p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="font-semibold">{friendlyType(activity.type)}</p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {activityDate(activity.startedAt)}
                              {participantNames.length > 1
                                ? ` · ${participantNames.join(' + ')}`
                                : ''}
                            </p>
                          </div>
                          <span className="flex shrink-0 items-center gap-1 rounded-full bg-background/70 px-2.5 py-1 text-xs font-semibold text-muted-foreground">
                            <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
                            {minutes === null ? 'In progress' : `${minutes} min`}
                          </span>
                        </div>
                      </Card>
                    );
                  })}

                  {activities.hasNextPage && (
                    <Button
                      variant="outline"
                      className="w-full bg-transparent"
                      disabled={activities.isFetchingNextPage}
                      onClick={() => activities.fetchNextPage()}
                    >
                      {activities.isFetchingNextPage && (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                      )}
                      Load older activity
                    </Button>
                  )}
                </div>
              )}
            </section>
          </>
        ) : null}
      </main>

      <BottomNav />
    </div>
  );
}
