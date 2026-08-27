'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Brain, CheckCircle2, Gamepad2, Loader2, PawPrint, Share2, Sparkles, TimerReset } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  socialAdventureApi,
  type ArcadeAttempt,
  type ArcadeReceipt,
  type ArcadeScenario,
} from '@/lib/api/social-adventure';

export default function ArcadePage() {
  const queryClient = useQueryClient();
  const [attempt, setAttempt] = useState<ArcadeAttempt | null>(null);
  const [receipt, setReceipt] = useState<ArcadeReceipt | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const timingStartRef = useRef<number | null>(null);

  const catalog = useQuery({
    queryKey: ['social-adventure', 'arcade'],
    queryFn: socialAdventureApi.arcade,
    retry: false,
  });

  const startMutation = useMutation({
    mutationFn: (challengeKey: ArcadeScenario['challengeKey']) =>
      socialAdventureApi.startArcadeAttempt(challengeKey),
    onSuccess: (result) => {
      setAttempt(result);
      setReceipt(null);
      setElapsedMs(0);
      timingStartRef.current = result.scenario.timing ? performance.now() : null;
    },
  });

  const completeMutation = useMutation({
    mutationFn: ({ attemptId, response }: { attemptId: string; response: Record<string, unknown> }) =>
      socialAdventureApi.completeArcadeAttempt(attemptId, response),
    onSuccess: async (result) => {
      setReceipt(result);
      timingStartRef.current = null;
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'arcade'] }),
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'me'] }),
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'leaderboard'] }),
      ]);
    },
  });

  const shareMutation = useMutation({
    mutationFn: (attemptId: string) =>
      socialAdventureApi.createShare({
        sourceType: 'HUMAN_SKILL_ATTEMPT',
        sourceId: attemptId,
        visibility: 'PUBLIC',
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['social-adventure', 'feed'] }),
  });

  useEffect(() => {
    if (!attempt?.scenario.timing || receipt || timingStartRef.current === null) return;
    const timer = window.setInterval(() => {
      if (timingStartRef.current === null) return;
      const elapsed = performance.now() - timingStartRef.current;
      setElapsedMs(Math.min(attempt.scenario.timing?.durationMs ?? elapsed, elapsed));
    }, 40);
    return () => window.clearInterval(timer);
  }, [attempt, receipt]);

  const finishChoice = (optionId: string) => {
    if (!attempt || completeMutation.isPending) return;
    completeMutation.mutate({ attemptId: attempt.attemptId, response: { optionId } });
  };

  const markTiming = () => {
    if (!attempt || timingStartRef.current === null || completeMutation.isPending) return;
    const tapMs = performance.now() - timingStartRef.current;
    completeMutation.mutate({ attemptId: attempt.attemptId, response: { tapMs } });
  };

  const resetRound = () => {
    setAttempt(null);
    setReceipt(null);
    setElapsedMs(0);
    timingStartRef.current = null;
    completeMutation.reset();
    shareMutation.reset();
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Gamepad2 className="h-5 w-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Human Skill Arcade
            </p>
            <h1 className="text-lg font-bold tracking-tight">Train your side of the leash</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.11] via-card/95 to-secondary/[0.06] p-5">
          <p className="eyebrow">The dog gets the day off</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight">Practice the human mechanics as a game.</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Timing, shaping decisions, catching useful behavior, and positive pairing are scored here.
            Only your best score in each game counts this week, so grinding repetitions gives no advantage.
          </p>
          <Button variant="outline" asChild className="mt-4 bg-transparent">
            <Link href="/community">See Community league →</Link>
          </Button>
        </section>

        {catalog.isLoading ? (
          <div className="flex min-h-48 items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
          </div>
        ) : catalog.error || !catalog.data ? (
          <div className="surface-soft mt-5 rounded-2xl p-5 text-center">
            <Brain className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
            <p className="mt-2 font-semibold">Arcade is unavailable right now.</p>
          </div>
        ) : !attempt ? (
          <section className="mt-6 space-y-3">
            {catalog.data.challenges.map((challenge) => (
              <article key={challenge.challengeKey} className="surface-soft rounded-3xl p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <p className="eyebrow">{challenge.skill}</p>
                    <h2 className="mt-1 text-lg font-bold">{challenge.title}</h2>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                      {challenge.prompt}
                    </p>
                  </div>
                  <span className="shrink-0 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                    {challenge.bestScore === null ? 'New' : `Best ${challenge.bestScore}`}
                  </span>
                </div>
                <Button
                  className="mt-4"
                  disabled={startMutation.isPending}
                  onClick={() => startMutation.mutate(challenge.challengeKey)}
                >
                  {startMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Play round
                </Button>
              </article>
            ))}
          </section>
        ) : (
          <section className="mt-6 rounded-3xl border border-border/70 bg-card/75 p-5">
            <div className="flex items-start gap-3">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
                <Sparkles className="h-5 w-5" aria-hidden="true" />
              </span>
              <div>
                <p className="eyebrow">{attempt.scenario.skill}</p>
                <h2 className="mt-1 text-xl font-bold">{attempt.scenario.title}</h2>
              </div>
            </div>

            <p className="mt-4 text-base leading-relaxed">{attempt.scenario.prompt}</p>

            {!receipt && attempt.scenario.options && (
              <div className="mt-5 grid gap-2">
                {attempt.scenario.options.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    disabled={completeMutation.isPending}
                    onClick={() => finishChoice(option.id)}
                    className="rounded-2xl border border-border bg-background/50 p-4 text-left text-sm font-semibold leading-relaxed transition-colors hover:border-primary/40 hover:bg-primary/[0.05] disabled:opacity-50"
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}

            {!receipt && attempt.scenario.timing && (
              <div className="mt-5 rounded-2xl bg-background/55 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold">
                  <TimerReset className="h-4 w-4 text-primary" aria-hidden="true" />
                  Watch the behavior track
                </div>
                <Progress
                  className="mt-4 h-3"
                  value={(elapsedMs / attempt.scenario.timing.durationMs) * 100}
                />
                <div className="mt-3 min-h-12 rounded-xl border border-border/70 bg-card/70 p-3 text-center text-sm">
                  {elapsedMs < attempt.scenario.timing.targetAtMs - 700
                    ? 'Approaching the mat…'
                    : elapsedMs < attempt.scenario.timing.targetAtMs
                      ? 'Almost there…'
                      : elapsedMs < attempt.scenario.timing.targetAtMs + 350
                        ? `Target: ${attempt.scenario.timing.targetLabel}`
                        : 'Behavior moved on'}
                </div>
                <Button size="lg" className="mt-4 w-full" onClick={markTiming}>
                  Mark now
                </Button>
              </div>
            )}

            {completeMutation.isPending && (
              <div className="mt-4 flex items-center justify-center gap-2 text-sm text-muted-foreground" role="status">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                Scoring the human move…
              </div>
            )}

            {receipt && (
              <div className="mt-5 rounded-2xl bg-primary/10 p-5">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="mt-0.5 h-6 w-6 shrink-0 text-primary" aria-hidden="true" />
                  <div>
                    <p className="eyebrow">Round complete</p>
                    <p className="mt-1 text-3xl font-black text-primary">{receipt.score}/100</p>
                    {receipt.timingErrorMs !== undefined && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        {receipt.timingErrorMs} ms from the target moment
                      </p>
                    )}
                  </div>
                </div>
                <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                  {receipt.explanation}
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button onClick={resetRound}>Another game</Button>
                  <Button
                    variant="outline"
                    className="bg-transparent"
                    disabled={shareMutation.isPending || shareMutation.isSuccess}
                    onClick={() => shareMutation.mutate(receipt.attemptId)}
                  >
                    <Share2 className="mr-2 h-4 w-4" aria-hidden="true" />
                    {shareMutation.isSuccess ? 'Shared' : 'Share result'}
                  </Button>
                </div>
                <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
                  Sharing publishes this Human Skill result, not a pet score or private dog history.
                </p>
              </div>
            )}
          </section>
        )}

        <section className="mt-6 rounded-3xl border border-border/60 bg-card/55 p-5">
          <div className="flex items-start gap-3">
            <PawPrint className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-bold">Arcade success is not dog-training authority</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                These games teach general reward-based mechanics. Significant fear, aggression, pain,
                or sudden behavior change should not become a DIY exposure level; Woof should route
                those situations toward qualified professional or veterinary help.
              </p>
            </div>
          </div>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
