'use client';

import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  Bookmark,
  Camera,
  Check,
  Clock3,
  EyeOff,
  Footprints,
  Image as ImageIcon,
  Loader2,
  Map,
  MapPin,
  PawPrint,
  RotateCcw,
  Sparkles,
  Star,
} from 'lucide-react';
import Link from 'next/link';
import { useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { storyApi, type StoryMoment, type StorySourceType } from '@/lib/api/story';

const sourceLabels: Record<StorySourceType, string> = {
  ACTIVITY: 'Activity',
  CARE_EVENT: 'Care',
  MEDIA: 'Memory',
};

function formatDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Unknown date';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === new Date().getFullYear() ? undefined : 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

function sourceIcon(moment: StoryMoment) {
  if (moment.sourceType === 'MEDIA') return Camera;
  if (moment.sourceType === 'ACTIVITY') return Footprints;
  if (moment.kind === 'TRACKER_DAILY_ACTIVITY') return Activity;
  return PawPrint;
}

export default function JourneyPage() {
  const queryClient = useQueryClient();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [noteDraft, setNoteDraft] = useState('');

  const story = useInfiniteQuery({
    queryKey: ['story'],
    queryFn: ({ pageParam }) =>
      storyApi.get({
        ...(pageParam ? { before: pageParam } : {}),
        limit: 36,
      }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.nextBefore ?? undefined,
    retry: false,
  });

  const dashboard = story.data?.pages[0];
  const moments = useMemo(() => {
    const byId = new Map<string, StoryMoment>();
    for (const page of story.data?.pages ?? []) {
      for (const moment of page.moments) byId.set(moment.id, moment);
    }
    return [...byId.values()].sort(
      (a, b) => new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime()
    );
  }, [story.data]);

  const curate = useMutation({
    mutationFn: storyApi.curate,
    onSuccess: async () => {
      setEditingId(null);
      setNoteDraft('');
      await queryClient.invalidateQueries({ queryKey: ['story'] });
    },
  });

  const beginNote = (moment: StoryMoment) => {
    setEditingId(moment.id);
    setNoteDraft(moment.curation.note ?? '');
  };

  const saveMoment = (moment: StoryMoment, note?: string) =>
    curate.mutate({
      sourceType: moment.sourceType,
      sourceId: moment.sourceId,
      action: 'SAVE',
      ...(note?.trim() ? { note: note.trim() } : {}),
    });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
            <Map className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              dogOS life record
            </p>
            <h1 className="text-lg font-bold tracking-tight">Our Story</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.1] via-card/90 to-secondary/[0.07] p-5 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Sparkles className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">A life, not a feed</p>
              <h2 className="mt-1 text-2xl font-bold tracking-tight">
                The little days are becoming a story.
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Activities, care moments, private memories, and bounded tracker context meet here in
                chronological order. Story references the real source records instead of copying
                them into a second truth.
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button asChild>
                  <Link href="/activity">Do something together</Link>
                </Button>
                <Button variant="outline" asChild className="bg-transparent">
                  <Link href="/library">
                    <Camera className="mr-2 h-4 w-4" aria-hidden="true" />
                    Media Library
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>

        {story.isLoading ? (
          <div className="flex min-h-[45vh] items-center justify-center" role="status">
            <div className="text-center">
              <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
              <p className="mt-3 text-sm text-muted-foreground">Turning the pages…</p>
            </div>
          </div>
        ) : story.isError || !dashboard ? (
          <section className="surface-soft mt-4 rounded-3xl p-6 text-center">
            <PawPrint className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h2 className="mt-3 text-lg font-bold">Our Story is unavailable</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              This surface may be paused for the current environment. Your Activity, Adventure, and
              Media Library records are unchanged.
            </p>
          </section>
        ) : (
          <>
            <section className="mt-6">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Life stats</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">Built from recorded life</h2>
                </div>
                {dashboard.stats.coverage === 'BOUNDED' && (
                  <span className="rounded-full border border-border px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    Recent-history estimate
                  </span>
                )}
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
                <Stat icon={Footprints} value={dashboard.stats.activities} label="Activities" />
                <Stat
                  icon={Clock3}
                  value={(dashboard.stats.activeMinutes / 60).toFixed(1)}
                  label="Hours"
                />
                <Stat
                  icon={Map}
                  value={(dashboard.stats.distanceMeters / 1000).toFixed(1)}
                  label="Kilometers"
                />
                <Stat icon={ImageIcon} value={dashboard.stats.memories} label="Memories" />
                <Stat icon={MapPin} value={dashboard.stats.namedPlaces} label="Named places" />
              </div>
            </section>

            {dashboard.milestones.length > 0 && (
              <section className="mt-7">
                <p className="eyebrow">Milestones</p>
                <h2 className="mt-1 text-xl font-bold tracking-tight">Quiet landmarks</h2>
                <div className="mt-3 flex gap-3 overflow-x-auto pb-2">
                  {dashboard.milestones.map((milestone) => (
                    <article key={milestone.id} className="surface-soft min-w-[210px] rounded-2xl p-4">
                      <Star className="h-5 w-5 text-primary" aria-hidden="true" />
                      <h3 className="mt-3 font-semibold">{milestone.title}</h3>
                      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                        {milestone.description}
                      </p>
                      <p className="mt-3 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                        {formatDate(milestone.achievedAt)}
                      </p>
                    </article>
                  ))}
                </div>
              </section>
            )}

            <section className="mt-7">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Timeline</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">What you have lived</h2>
                </div>
                <span className="text-xs text-muted-foreground">{moments.length} shown</span>
              </div>

              {moments.length === 0 ? (
                <div className="surface-soft mt-3 flex min-h-44 flex-col items-center justify-center rounded-3xl p-6 text-center">
                  <PawPrint className="h-7 w-7 text-primary" aria-hidden="true" />
                  <p className="mt-3 font-semibold">The first page is still blank</p>
                  <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                    A walk, a quiet recovery choice, or a kept photo can become the first real
                    moment. Nothing here requires posting publicly.
                  </p>
                </div>
              ) : (
                <div className="mt-3 space-y-3">
                  {moments.map((moment) => {
                    const Icon = sourceIcon(moment);
                    const isSaved = moment.curation.state === 'SAVED';
                    const isEditing = editingId === moment.id;
                    return (
                      <article key={moment.id} className="surface-soft rounded-3xl p-4">
                        <div className="flex items-start gap-3">
                          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                            <Icon className="h-4.5 w-4.5" aria-hidden="true" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <span className="rounded-full border border-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                {sourceLabels[moment.sourceType]}
                              </span>
                              {moment.suggested && !isSaved && (
                                <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                                  Worth remembering?
                                </span>
                              )}
                              {isSaved && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                                  <Check className="h-3 w-3" aria-hidden="true" /> Saved
                                </span>
                              )}
                            </div>
                            <h3 className="mt-2 font-semibold">{moment.title}</h3>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {moment.petNames.join(' + ') || 'Household'} · {formatDate(moment.occurredAt)}
                            </p>
                            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                              {moment.summary}
                            </p>

                            {moment.curation.note && !isEditing && (
                              <blockquote className="mt-3 rounded-2xl border-l-2 border-primary/40 bg-background/55 px-3 py-2 text-sm italic text-muted-foreground">
                                {moment.curation.note}
                              </blockquote>
                            )}

                            {isEditing && (
                              <div className="mt-3 rounded-2xl border border-border bg-background/60 p-3">
                                <label className="text-xs font-semibold text-muted-foreground">
                                  Your note
                                  <textarea
                                    className="mt-1.5 min-h-20 w-full resize-y rounded-xl border border-border bg-background p-3 text-sm outline-none transition focus:border-primary/60 focus:ring-2 focus:ring-primary/10"
                                    maxLength={500}
                                    value={noteDraft}
                                    placeholder="What makes this one worth keeping?"
                                    onChange={(event) => setNoteDraft(event.target.value)}
                                  />
                                </label>
                                <div className="mt-2 flex gap-2">
                                  <Button
                                    size="sm"
                                    disabled={curate.isPending}
                                    onClick={() => saveMoment(moment, noteDraft)}
                                  >
                                    Save note
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => {
                                      setEditingId(null);
                                      setNoteDraft('');
                                    }}
                                  >
                                    Cancel
                                  </Button>
                                </div>
                              </div>
                            )}

                            <div className="mt-3 flex flex-wrap gap-2">
                              {!isSaved ? (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  disabled={curate.isPending}
                                  onClick={() => saveMoment(moment)}
                                >
                                  <Bookmark className="mr-1.5 h-4 w-4" aria-hidden="true" />
                                  Save
                                </Button>
                              ) : (
                                <>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    disabled={curate.isPending}
                                    onClick={() => beginNote(moment)}
                                  >
                                    Add note
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    disabled={curate.isPending}
                                    onClick={() =>
                                      curate.mutate({
                                        sourceType: moment.sourceType,
                                        sourceId: moment.sourceId,
                                        action: 'CLEAR',
                                      })
                                    }
                                  >
                                    <RotateCcw className="mr-1.5 h-4 w-4" aria-hidden="true" />
                                    Unsave
                                  </Button>
                                </>
                              )}
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={curate.isPending}
                                onClick={() =>
                                  curate.mutate({
                                    sourceType: moment.sourceType,
                                    sourceId: moment.sourceId,
                                    action: 'HIDE',
                                  })
                                }
                              >
                                <EyeOff className="mr-1.5 h-4 w-4" aria-hidden="true" />
                                Hide from Story
                              </Button>
                              {moment.sourceType === 'MEDIA' && (
                                <Button size="sm" variant="ghost" asChild>
                                  <Link href="/library">Open memory</Link>
                                </Button>
                              )}
                            </div>
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}

              {story.hasNextPage && (
                <div className="mt-4 text-center">
                  <Button
                    variant="outline"
                    disabled={story.isFetchingNextPage}
                    onClick={() => story.fetchNextPage()}
                  >
                    {story.isFetchingNextPage && (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                    )}
                    Read older pages
                  </Button>
                </div>
              )}
            </section>

            <section className="mt-7 rounded-3xl border border-primary/15 bg-primary/[0.04] p-5">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div>
                  <p className="font-semibold">Your sources stay themselves</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Saving or hiding a Story moment changes only your Story curation. It does not
                    edit or delete the Activity, CareEvent, tracker observation, or Media Library
                    asset underneath it. Raw tracker GPS is not part of this Story surface.
                  </p>
                </div>
              </div>
            </section>
          </>
        )}
      </main>

      <BottomNav />
    </div>
  );
}

function Stat({
  icon: Icon,
  value,
  label,
}: {
  icon: typeof PawPrint;
  value: string | number;
  label: string;
}) {
  return (
    <div className="surface-soft rounded-2xl p-3">
      <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
      <p className="mt-3 text-lg font-bold tracking-tight">{value}</p>
      <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
    </div>
  );
}
