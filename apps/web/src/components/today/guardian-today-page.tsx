'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Brain,
  Check,
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
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { ConciergeBriefing } from '@/components/concierge/concierge-briefing';
import { PetSwitcher } from '@/components/pets/pet-switcher';
import { RelationshipTools } from '@/components/today/relationship-tools';
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

type CompletionReceipt = {
  message: string;
  rewardCopy: string;
  rewardExplanation: string;
  duplicate: boolean;
};

function QuestActions({
  quest,
  startingQuestId,
  onStart,
  onComplete,
  onSafeStop,
  prominent = false,
}: {
  quest: AdventureQuest;
  startingQuestId: string | null;
  onStart: (quest: AdventureQuest) => void;
  onComplete: (quest: AdventureQuest) => void;
  onSafeStop: (quest: AdventureQuest) => void;
  prominent?: boolean;
}) {
  return (
    <div className={prominent ? 'mt-5 space-y-2' : 'mt-3 flex flex-wrap gap-2'}>
      <Button
        size={prominent ? 'lg' : 'sm'}
        className={prominent ? 'w-full sm:w-auto' : undefined}
        disabled={startingQuestId !== null}
        onClick={() => onStart(quest)}
      >
        {startingQuestId === quest.id && (
          <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden="true" />
        )}
        {quest.actionLabel}
      </Button>
      <div className={prominent ? 'flex flex-wrap gap-2' : 'contents'}>
        <Button
          variant="outline"
          size="sm"
          className="bg-transparent"
          onClick={() => onComplete(quest)}
        >
          <Check className="mr-1.5 h-4 w-4" aria-hidden="true" />
          Close the loop
        </Button>
        {quest.safeStopEligible && (
          <Button variant="ghost" size="sm" onClick={() => onSafeStop(quest)}>
            I listened and stopped
          </Button>
        )}
      </div>
    </div>
  );
}

export default function HomePage() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const [closingQuest, setClosingQuest] = useState<AdventureQuest | null>(null);
  const [startingQuestId, setStartingQuestId] = useState<string | null>(null);
  const [dogExperience, setDogExperience] = useState<
    'loved_it' | 'comfortable' | 'not_their_thing' | null
  >(null);
  const [safeOptOut, setSafeOptOut] = useState(false);
  const [completionReceipt, setCompletionReceipt] = useState<CompletionReceipt | null>(null);

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
      const rewardCopy = result.reward.duplicate
        ? 'Already saved · no additional Bond XP'
        : result.reward.bondXp > 0
          ? `+${result.reward.bondXp} Bond XP`
          : 'No Bond XP this time';
      setCompletionReceipt({
        message: result.message,
        rewardCopy,
        rewardExplanation: result.reward.explanation,
        duplicate: result.reward.duplicate,
      });
      await queryClient.invalidateQueries({ queryKey: ['adventure', 'me'] });
    },
  });

  const startQuest = async (quest: AdventureQuest) => {
    if (!data || startingQuestId) return;

    setStartingQuestId(quest.id);
    try {
      await adventureApi.selectQuest(quest.id, data.pet.id);
    } finally {
      // Selection persistence improves continuity, but a transient analytics/network
      // failure must never trap the user on Today instead of letting them do the activity.
      setStartingQuestId(null);
      router.push(quest.href);
    }
  };

  const closeOutcome = () => {
    setClosingQuest(null);
    setDogExperience(null);
    setSafeOptOut(false);
    setCompletionReceipt(null);
    completeMutation.reset();
  };

  const openCompletion = (quest: AdventureQuest, optOut = false) => {
    setClosingQuest(quest);
    setCompletionReceipt(null);
    setSafeOptOut(optOut);
    setDogExperience(optOut ? 'not_their_thing' : null);
  };

  const bestQuest = data?.quests[0] ?? null;
  const alternatives = data?.quests.slice(1) ?? [];

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center px-4">
          <Link href="/" className="flex items-center gap-3" aria-label="Woof Today">
            <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
              <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Dog + human
              </span>
              <span className="block text-lg font-bold tracking-tight">Today</span>
            </span>
          </Link>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        {isLoading ? (
          <div className="flex min-h-[60vh] items-center justify-center" role="status">
            <div className="text-center">
              <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
              <p className="mt-3 text-sm text-muted-foreground">
                Finding one good thing to do together…
              </p>
            </div>
          </div>
        ) : error || !data ? (
          <section className="surface-soft rounded-3xl p-6 text-center">
            <PawPrint className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h1 className="mt-3 text-xl font-bold">Adventure mode is unavailable</h1>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Adventure may be paused or temporarily unavailable. Your existing Coach, Health,
              Library, and social tools are still available.
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
            <PetSwitcher currentPetId={data.pet.id} label="Today is for" withDivider={false} />

            {bestQuest ? (
              <section
                data-today-primary-quest
                className="mt-4 rounded-3xl border border-primary/25 bg-gradient-to-br from-primary/[0.12] via-card/95 to-secondary/[0.06] p-5 shadow-sm sm:p-6"
                aria-labelledby="today-primary-quest-heading"
              >
                <div className="flex items-start gap-4">
                  <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
                    {(() => {
                      const Icon = pathwayIcons[bestQuest.primaryPathway];
                      return <Icon className="h-6 w-6" aria-hidden="true" />;
                    })()}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="eyebrow">A good place to start with {data.pet.name}</p>
                    <h1
                      id="today-primary-quest-heading"
                      className="mt-1 text-3xl font-bold tracking-tight text-balance"
                    >
                      {bestQuest.title}
                    </h1>
                  </div>
                </div>

                <p className="mt-4 text-base leading-relaxed text-foreground/90">
                  {bestQuest.description}
                </p>

                <div className="mt-4 rounded-2xl border border-border/70 bg-background/55 p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
                    Why this one today
                  </p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    {bestQuest.why}
                  </p>
                </div>

                <QuestActions
                  quest={bestQuest}
                  startingQuestId={startingQuestId}
                  onStart={(quest) => void startQuest(quest)}
                  onComplete={(quest) => openCompletion(quest)}
                  onSafeStop={(quest) => openCompletion(quest, true)}
                  prominent
                />

                <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
                  Woof recommends, you choose. Changing your mind, making it easier, or stopping
                  when
                  {` ${data.pet.name}`} is done can all be the right outcome.
                </p>
              </section>
            ) : (
              <section className="mt-4 surface-soft rounded-3xl p-6 text-center">
                <PawPrint className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
                <h1 className="mt-3 text-xl font-bold">Nothing needs pushing today</h1>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  Woof does not have a useful quest to recommend right now. Rest or your usual
                  routine is a valid choice.
                </p>
              </section>
            )}

            {alternatives.length > 0 && (
              <details
                data-today-alternatives
                className="mt-4 rounded-2xl border border-border/70 bg-card/65 px-4 py-3"
              >
                <summary className="cursor-pointer text-sm font-semibold">
                  Want a different kind of day?{' '}
                  <span className="font-normal text-muted-foreground">
                    {alternatives.length} other {alternatives.length === 1 ? 'option' : 'options'}
                  </span>
                </summary>
                <div className="mt-3 space-y-3 border-t border-border/60 pt-3">
                  {alternatives.map((quest) => {
                    const Icon = pathwayIcons[quest.primaryPathway];
                    return (
                      <article key={quest.id} className="rounded-2xl bg-background/50 p-4">
                        <div className="flex items-start gap-3">
                          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                            <Icon className="h-4 w-4" aria-hidden="true" />
                          </span>
                          <div className="min-w-0 flex-1">
                            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                              {quest.variant === 'wildcard' ? 'Something different' : 'Alternative'}{' '}
                              · {quest.primaryPathway.toLowerCase()}
                            </p>
                            <h2 className="mt-1 font-bold tracking-tight">{quest.title}</h2>
                            <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                              {quest.description}
                            </p>
                            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                              {quest.why}
                            </p>
                            <QuestActions
                              quest={quest}
                              startingQuestId={startingQuestId}
                              onStart={(candidate) => void startQuest(candidate)}
                              onComplete={(candidate) => openCompletion(candidate)}
                              onSafeStop={(candidate) => openCompletion(candidate, true)}
                            />
                          </div>
                        </div>
                      </article>
                    );
                  })}
                </div>
              </details>
            )}

            {data.learningSummary.length > 0 && (
              <section
                data-today-learning
                className="mt-6 rounded-3xl border border-secondary/20 bg-secondary/[0.05] p-5"
              >
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

            <details
              data-today-concierge
              className="mt-6 rounded-2xl border border-border/70 bg-card/50 px-4 py-3"
            >
              <summary className="cursor-pointer text-sm font-semibold">
                More context for today{' '}
                <span className="font-normal text-muted-foreground">from Concierge</span>
              </summary>
              <div className="mt-4 border-t border-border/60 pt-4">
                <ConciergeBriefing showPetSwitcher={false} />
              </div>
            </details>

            <RelationshipTools />

            <section data-today-progress className="mt-7" aria-labelledby="recent-rhythm-heading">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Recent rhythm</p>
                  <h2 id="recent-rhythm-heading" className="mt-1 text-lg font-bold tracking-tight">
                    Context, not today&apos;s assignment
                  </h2>
                </div>
                <Link href="/compass" className="text-sm font-semibold text-primary">
                  Full compass →
                </Link>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-3">
                <div className="surface-soft rounded-2xl p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    Bond XP
                  </p>
                  <p className="mt-1 text-xl font-bold text-primary">{data.bondXp}</p>
                  <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                    Game progress, not a wellbeing score.
                  </p>
                </div>
                <div className="surface-soft rounded-2xl p-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    {data.rhythm.label}
                  </p>
                  <p className="mt-1 text-xl font-bold text-primary">
                    {data.rhythm.activeWeeks}/{data.rhythm.windowWeeks}
                  </p>
                  <Progress
                    className="mt-2 h-1.5"
                    value={(data.rhythm.activeWeeks / data.rhythm.windowWeeks) * 100}
                  />
                </div>
              </div>

              {data.compass.length > 0 && (
                <details className="mt-3 rounded-2xl border border-border/70 bg-background/40 px-4 py-3">
                  <summary className="cursor-pointer text-sm font-semibold">
                    See recent Pawprint Compass opportunities
                  </summary>
                  <div className="mt-3 grid grid-cols-2 gap-3 border-t border-border/60 pt-3">
                    {data.compass.slice(0, 4).map((item) => {
                      const Icon = pathwayIcons[item.pathway];
                      return (
                        <div key={item.pathway} className="surface-soft rounded-2xl p-4">
                          <div className="flex items-center justify-between gap-2">
                            <span className="flex items-center gap-2 text-sm font-semibold">
                              <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
                              {item.label}
                            </span>
                            <span className="text-xs font-bold text-primary">
                              {item.recentDays}d
                            </span>
                          </div>
                          <Progress className="mt-3 h-1.5" value={item.coverage} />
                          <p className="mt-2 text-[11px] text-muted-foreground">
                            Recent opportunity coverage · 28-day window
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </details>
              )}
            </section>

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

            {completionReceipt ? (
              <div className="mt-5 rounded-2xl bg-primary/10 p-5">
                <div className="flex items-start gap-3">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
                    <PawPrint className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div className="min-w-0">
                    <p className="eyebrow">What Woof learned</p>
                    <p className="mt-1 font-semibold leading-relaxed">
                      {completionReceipt.message}
                    </p>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      {completionReceipt.duplicate
                        ? 'This outcome was already in your shared history, so Woof did not count it twice.'
                        : `This outcome is now part of ${data.pet.name}'s recent shared pattern and can influence future quest ranking.`}
                    </p>
                  </div>
                </div>

                <div className="mt-4 rounded-xl border border-border/70 bg-background/55 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                      Game progress
                    </p>
                    <span className="text-xs font-bold text-primary">
                      {completionReceipt.rewardCopy}
                    </span>
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                    {completionReceipt.rewardExplanation}
                  </p>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
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
