'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Brain,
  Check,
  Compass,
  Footprints,
  Heart,
  HeartHandshake,
  Loader2,
  MoonStar,
  PawPrint,
  ShieldCheck,
  Sparkles,
  TreePine,
  X,
} from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { adventureApi, type AdventureQuest, type WellbeingPathway } from '@/lib/api/adventure';

const pathwayIcons: Record<WellbeingPathway, typeof PawPrint> = {
  MOVE: Footprints,
  EXPLORE: TreePine,
  ENRICH: Sparkles,
  LEARN: Brain,
  CONNECT: HeartHandshake,
  CARE: ShieldCheck,
  RECOVER: MoonStar,
  BOND: Heart,
};

const dogChoices = [
  { value: 'loved_it' as const, emoji: '😄', label: 'Loved it' },
  { value: 'comfortable' as const, emoji: '😌', label: 'Comfortable' },
  { value: 'not_their_thing' as const, emoji: '😕', label: 'Not their thing' },
];

const ownerChoices = [
  { value: 'great' as const, emoji: '✨', label: 'Great' },
  { value: 'fine' as const, emoji: '🙂', label: 'Fine' },
  { value: 'a_lot_today' as const, emoji: '😮‍💨', label: 'A lot today' },
];

export default function HomePage() {
  const queryClient = useQueryClient();
  const [closingQuest, setClosingQuest] = useState<AdventureQuest | null>(null);
  const [dogExperience, setDogExperience] = useState<
    'loved_it' | 'comfortable' | 'not_their_thing' | null
  >(null);
  const [safeOptOut, setSafeOptOut] = useState(false);
  const [completionMessage, setCompletionMessage] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['adventure', 'me'],
    queryFn: () => adventureApi.getMine(),
    retry: false,
  });

  const completeMutation = useMutation({
    mutationFn: ({
      quest,
      ownerExperience,
    }: {
      quest: AdventureQuest;
      ownerExperience: 'great' | 'fine' | 'a_lot_today';
    }) => {
      if (!data || !dogExperience) throw new Error('Outcome is incomplete');
      return adventureApi.completeQuest(quest.id, {
        petId: data.pet.id,
        dogExperience,
        ownerExperience,
        safeOptOut,
      });
    },
    onSuccess: async (result) => {
      setCompletionMessage(
        `${result.message} ${result.reward.bondXp > 0 ? `+${result.reward.bondXp} Bond XP` : ''}`.trim()
      );
      await queryClient.invalidateQueries({ queryKey: ['adventure', 'me'] });
    },
  });

  const closeOutcome = () => {
    setClosingQuest(null);
    setDogExperience(null);
    setSafeOptOut(false);
    setCompletionMessage(null);
    completeMutation.reset();
  };

  const openCompletion = (quest: AdventureQuest, optOut = false) => {
    setClosingQuest(quest);
    setCompletionMessage(null);
    setSafeOptOut(optOut);
    setDogExperience(optOut ? 'not_their_thing' : null);
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-3" aria-label="Woof Today">
            <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
              <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Dog + human
              </span>
              <span className="block text-lg font-bold tracking-tight">Woof Adventure</span>
            </span>
          </Link>
          {data && (
            <div className="rounded-2xl border border-border/60 bg-card/70 px-3 py-1.5 text-right">
              <p className="text-[9px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Bond XP
              </p>
              <p className="text-base font-bold text-primary">{data.bondXp}</p>
            </div>
          )}
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        {isLoading ? (
          <div className="flex min-h-[60vh] items-center justify-center" role="status">
            <div className="text-center">
              <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
              <p className="mt-3 text-sm text-muted-foreground">
                Building today&apos;s quest deck…
              </p>
            </div>
          </div>
        ) : error || !data ? (
          <section className="surface-soft rounded-3xl p-6 text-center">
            <PawPrint className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h1 className="mt-3 text-xl font-bold">Adventure mode is unavailable</h1>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Your existing Coach, Health, Library, and social tools are still available. The new
              quest ledger requires the latest database migration.
            </p>
            <Button
              className="mt-5"
              onClick={() => queryClient.invalidateQueries({ queryKey: ['adventure'] })}
            >
              Try again
            </Button>
          </section>
        ) : (
          <>
            <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.11] via-card/90 to-secondary/[0.08] p-5 shadow-sm">
              <p className="eyebrow">Today&apos;s party quest</p>
              <h1 className="mt-1 text-3xl font-bold tracking-tight">
                {data.pet.name} has {data.quests.length} adventures available.
              </h1>
              <p className="mt-2 max-w-lg text-sm leading-relaxed text-muted-foreground">
                Woof recommends. You choose. Rest, changing your mind, or listening when your dog
                says “not today” can all be the right play.
              </p>

              <div className="mt-5 flex items-center gap-3 rounded-2xl border border-border/60 bg-background/55 p-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Heart className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-semibold">{data.rhythm.label}</p>
                    <span className="text-xs font-semibold text-primary">
                      {data.rhythm.activeWeeks}/{data.rhythm.windowWeeks} weeks
                    </span>
                  </div>
                  <Progress
                    className="mt-2 h-1.5"
                    value={(data.rhythm.activeWeeks / data.rhythm.windowWeeks) * 100}
                  />
                </div>
              </div>
            </section>

            <section className="mt-5 space-y-3" aria-label="Today's quests">
              {data.quests.map((quest, index) => {
                const Icon = pathwayIcons[quest.primaryPathway];
                return (
                  <article
                    key={quest.id}
                    className={`surface-soft rounded-3xl p-5 ${index === 0 ? 'border-primary/30 bg-primary/[0.035]' : ''}`}
                  >
                    <div className="flex items-start gap-4">
                      <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
                              {index === 0
                                ? 'Best quest'
                                : quest.variant === 'wildcard'
                                  ? 'Wildcard'
                                  : 'Alternative'}{' '}
                              · {quest.primaryPathway.toLowerCase()}
                            </p>
                            <h2 className="mt-1 text-lg font-bold tracking-tight">{quest.title}</h2>
                          </div>
                          <span className="shrink-0 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                            Base {quest.xp} XP
                          </span>
                        </div>
                        <p className="mt-2 text-sm leading-relaxed">{quest.description}</p>
                        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                          {quest.why}
                        </p>

                        <div className="mt-4 flex flex-wrap gap-1.5">
                          {quest.pathways.map((pathway) => (
                            <span
                              key={pathway}
                              className="rounded-full border border-border/70 bg-background/60 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground"
                            >
                              {pathway}
                            </span>
                          ))}
                        </div>

                        <div className="mt-4 flex flex-wrap gap-2">
                          <Button asChild size="sm">
                            <Link
                              href={quest.href}
                              onClick={() => void adventureApi.selectQuest(quest.id, data.pet.id)}
                            >
                              {quest.actionLabel}
                            </Link>
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="bg-transparent"
                            onClick={() => openCompletion(quest)}
                          >
                            <Check className="mr-1.5 h-4 w-4" aria-hidden="true" />
                            Close the loop
                          </Button>
                          {quest.safeStopEligible && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => openCompletion(quest, true)}
                            >
                              I listened and stopped
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </section>

            <section className="mt-7">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Pawprint Compass</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">
                    Recent opportunities, not a score
                  </h2>
                </div>
                <Link href="/compass" className="text-sm font-semibold text-primary">
                  Full compass →
                </Link>
              </div>
              <div className="mt-3 grid grid-cols-2 gap-3">
                {data.compass.slice(0, 4).map((item) => {
                  const Icon = pathwayIcons[item.pathway];
                  return (
                    <div key={item.pathway} className="surface-soft rounded-2xl p-4">
                      <div className="flex items-center justify-between gap-2">
                        <span className="flex items-center gap-2 text-sm font-semibold">
                          <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
                          {item.label}
                        </span>
                        <span className="text-xs font-bold text-primary">{item.recentDays}d</span>
                      </div>
                      <Progress className="mt-3 h-1.5" value={item.coverage} />
                      <p className="mt-2 text-[11px] text-muted-foreground">
                        {item.xp} pathway XP · 28-day window
                      </p>
                    </div>
                  );
                })}
              </div>
            </section>

            {data.learningSummary.length > 0 && (
              <section className="mt-6 rounded-3xl border border-secondary/20 bg-secondary/[0.05] p-5">
                <p className="eyebrow">What Woof is learning</p>
                <ul className="mt-3 space-y-2 text-sm leading-relaxed text-muted-foreground">
                  {data.learningSummary.slice(0, 3).map((line) => (
                    <li key={line} className="flex gap-2">
                      <Sparkles
                        className="mt-0.5 h-4 w-4 shrink-0 text-primary"
                        aria-hidden="true"
                      />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            <p className="mt-5 text-center text-xs leading-relaxed text-muted-foreground">
              {data.disclaimer}
            </p>
          </>
        )}
      </main>

      {closingQuest && data && (
        <div
          className="fixed inset-0 z-[70] flex items-end justify-center bg-background/70 p-3 backdrop-blur-sm sm:items-center"
          role="dialog"
          aria-modal="true"
          aria-label="Quest outcome"
        >
          <div className="w-full max-w-md rounded-3xl border border-border bg-card p-5 shadow-2xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="eyebrow">Five-second learning loop</p>
                <h2 className="mt-1 text-xl font-bold">{closingQuest.title}</h2>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={closeOutcome}
                aria-label="Close outcome flow"
              >
                <X className="h-5 w-5" aria-hidden="true" />
              </Button>
            </div>

            {completionMessage ? (
              <div className="mt-5 rounded-2xl bg-primary/10 p-5 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
                  <PawPrint className="h-6 w-6" aria-hidden="true" />
                </div>
                <p className="mt-3 font-semibold leading-relaxed">{completionMessage}</p>
                <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                  The result now informs future quest ranking for this dog-owner pair.
                </p>
                <div className="mt-4 flex justify-center gap-2">
                  <Button onClick={closeOutcome}>Done</Button>
                  <Button variant="outline" asChild className="bg-transparent">
                    <Link href="/library">Add a memory</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {safeOptOut ? (
                  <div className="mt-5 rounded-2xl border border-primary/20 bg-primary/[0.05] p-4">
                    <p className="font-semibold text-primary">You listened.</p>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                      Woof will treat giving space or ending the interaction as successful dog
                      literacy.
                    </p>
                  </div>
                ) : !dogExperience ? (
                  <div className="mt-5">
                    <p className="text-sm font-semibold">How was it for {data.pet.name}?</p>
                    <div className="mt-3 grid grid-cols-3 gap-2">
                      {dogChoices.map((choice) => (
                        <button
                          key={choice.value}
                          type="button"
                          onClick={() => setDogExperience(choice.value)}
                          className="rounded-2xl border border-border bg-background/55 p-3 text-center transition-colors hover:border-primary/40 hover:bg-primary/[0.05]"
                        >
                          <span className="block text-2xl" aria-hidden="true">
                            {choice.emoji}
                          </span>
                          <span className="mt-1 block text-xs font-semibold">{choice.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}

                {(dogExperience || safeOptOut) && (
                  <div className="mt-5">
                    <p className="text-sm font-semibold">How was it for you?</p>
                    <div className="mt-3 grid grid-cols-3 gap-2">
                      {ownerChoices.map((choice) => (
                        <button
                          key={choice.value}
                          type="button"
                          disabled={completeMutation.isPending}
                          onClick={() =>
                            completeMutation.mutate({
                              quest: closingQuest,
                              ownerExperience: choice.value,
                            })
                          }
                          className="rounded-2xl border border-border bg-background/55 p-3 text-center transition-colors hover:border-primary/40 hover:bg-primary/[0.05] disabled:opacity-50"
                        >
                          <span className="block text-2xl" aria-hidden="true">
                            {choice.emoji}
                          </span>
                          <span className="mt-1 block text-xs font-semibold">{choice.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {completeMutation.isPending && (
                  <div
                    className="mt-4 flex items-center justify-center gap-2 text-sm text-muted-foreground"
                    role="status"
                  >
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Learning from the outcome…
                  </div>
                )}
                {completeMutation.isError && (
                  <p className="mt-4 text-center text-sm text-destructive">
                    That outcome could not be saved. Nothing was lost from your existing history.
                  </p>
                )}
              </>
            )}
          </div>
        </div>
      )}

      <BottomNav />
    </div>
  );
}
