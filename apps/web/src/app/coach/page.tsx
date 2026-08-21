'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ArrowLeft,
  Brain,
  Check,
  ChevronRight,
  CircleAlert,
  Heart,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Timer,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import {
  type CoachingDashboard,
  type CoachingPlan,
  type CoachingProgression,
  type TrainingSessionInput,
  coachingApi,
} from '@/lib/api/coaching';
import { cn } from '@/lib/utils';

const rewards: Array<{ value: TrainingSessionInput['rewardType']; label: string }> = [
  { value: 'food', label: 'Food' },
  { value: 'play', label: 'Play' },
  { value: 'praise', label: 'Praise' },
  { value: 'access', label: 'Access' },
  { value: 'environmental', label: 'Sniff / explore' },
];

const observationOptions = [
  { value: 'look-away', label: 'Looking away' },
  { value: 'lip-lick', label: 'Lip licking' },
  { value: 'yawning', label: 'Yawning' },
  { value: 'panting', label: 'Panting' },
  { value: 'freezing', label: 'Freezing' },
  { value: 'escape-attempt', label: 'Trying to leave' },
  { value: 'growling', label: 'Growling' },
  { value: 'hiding', label: 'Hiding' },
  { value: 'tail-tucked', label: 'Tail tucked' },
];

const progressionTone: Record<CoachingProgression['action'], string> = {
  start: 'bg-primary/10 text-primary',
  hold: 'bg-primary/10 text-primary',
  increase: 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300',
  decrease: 'bg-amber-500/10 text-amber-800 dark:text-amber-300',
};

function ProgressionCard({ decision }: { decision: CoachingProgression }) {
  return (
    <div className="rounded-2xl border border-border/60 bg-card/70 p-4">
      <div className="flex items-start gap-3">
        <div
          className={cn(
            'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl',
            progressionTone[decision.action],
          )}
        >
          {decision.action === 'increase' ? (
            <Sparkles className="h-4 w-4" aria-hidden="true" />
          ) : decision.action === 'decrease' ? (
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          ) : (
            <Brain className="h-4 w-4" aria-hidden="true" />
          )}
        </div>
        <div>
          <p className="text-sm font-semibold">{decision.headline}</p>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{decision.reason}</p>
        </div>
      </div>
    </div>
  );
}

function PracticePanel({
  plan,
  onClose,
  onSaved,
}: {
  plan: CoachingPlan;
  onClose: () => void;
  onSaved: (decision: CoachingProgression) => void;
}) {
  const queryClient = useQueryClient();
  const [attempts, setAttempts] = useState(0);
  const [successes, setSuccesses] = useState(0);
  const [rewardType, setRewardType] = useState<TrainingSessionInput['rewardType']>('food');
  const [stressSignals, setStressSignals] = useState<string[]>([]);
  const [stoppedEarly, setStoppedEarly] = useState(false);
  const [startedAt] = useState(() => Date.now());

  const saveMutation = useMutation({
    mutationFn: () => {
      const durationSeconds = Math.max(
        20,
        Math.min(900, Math.round((Date.now() - startedAt) / 1000)),
      );
      return coachingApi.recordSession(plan.id, {
        attempts,
        successes,
        durationSeconds,
        distractionLevel: plan.level,
        rewardType,
        stressSignals,
        stoppedEarly,
      });
    },
    onSuccess: (result) => {
      void queryClient.invalidateQueries({ queryKey: ['coaching', 'me'] });
      onSaved(result.decision);
    },
  });

  const toggleObservation = (signal: string) => {
    setStressSignals((current) =>
      current.includes(signal)
        ? current.filter((candidate) => candidate !== signal)
        : [...current, signal].slice(0, 6),
    );
  };

  return (
    <div className="fixed inset-0 z-[70] overflow-y-auto bg-background">
      <div className="mx-auto min-h-screen max-w-xl px-4 pb-10 pt-4">
        <div className="flex items-center justify-between">
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close practice">
            <ArrowLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
          <span className="rounded-full border border-border/60 bg-card px-3 py-1 text-xs font-semibold text-muted-foreground">
            Level {plan.level} · {plan.levelLabel}
          </span>
        </div>

        <div className="mt-8 text-center">
          <p className="eyebrow">Practice together</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">{plan.title}</h1>
          <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-muted-foreground">
            Cue: <span className="font-semibold text-foreground">{plan.cue}</span>. Keep this short
            enough that both of you still want another round.
          </p>
        </div>

        <div className="mt-8 rounded-3xl border border-primary/15 bg-primary/[0.055] p-5">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
            Your job
          </p>
          <p className="mt-2 text-base font-semibold leading-relaxed">{plan.handlerFocus}</p>
        </div>

        <div className="mt-7 grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => {
              setAttempts((value) => value + 1);
              setSuccesses((value) => value + 1);
            }}
            className="flex min-h-32 flex-col items-center justify-center rounded-3xl border border-primary/20 bg-primary text-primary-foreground shadow-sm transition-transform active:scale-[0.98]"
          >
            <Check className="h-7 w-7" aria-hidden="true" />
            <span className="mt-2 text-lg font-bold">Nice!</span>
            <span className="mt-1 text-xs text-primary-foreground/75">Mark + reward</span>
          </button>
          <button
            type="button"
            onClick={() => setAttempts((value) => value + 1)}
            className="flex min-h-32 flex-col items-center justify-center rounded-3xl border border-border/70 bg-card transition-transform active:scale-[0.98]"
          >
            <RotateCcw className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
            <span className="mt-2 text-lg font-bold">Reset</span>
            <span className="mt-1 text-xs text-muted-foreground">Make the next rep easier</span>
          </button>
        </div>

        <div className="mt-4 flex items-center justify-center gap-6 text-sm">
          <span>
            <strong className="text-lg">{successes}</strong>{' '}
            <span className="text-muted-foreground">successes</span>
          </span>
          <span>
            <strong className="text-lg">{attempts}</strong>{' '}
            <span className="text-muted-foreground">attempts</span>
          </span>
        </div>

        <div className="mt-8">
          <p className="text-sm font-semibold">What was rewarding today?</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {rewards.map((reward) => (
              <button
                key={reward.value}
                type="button"
                onClick={() => setRewardType(reward.value)}
                className={cn(
                  'rounded-full border px-3 py-2 text-xs font-semibold transition-colors',
                  rewardType === reward.value
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border/70 bg-card text-muted-foreground',
                )}
              >
                {reward.label}
              </button>
            ))}
          </div>
        </div>

        <details className="mt-6 rounded-2xl border border-border/60 bg-card/60 p-4">
          <summary className="cursor-pointer text-sm font-semibold">
            Did anything look uncomfortable?
          </summary>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            These observations are context, not a diagnosis. Selecting one tells Woof to make the
            next setup easier.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {observationOptions.map((observation) => (
              <button
                key={observation.value}
                type="button"
                onClick={() => toggleObservation(observation.value)}
                className={cn(
                  'rounded-full border px-3 py-2 text-xs transition-colors',
                  stressSignals.includes(observation.value)
                    ? 'border-amber-500/40 bg-amber-500/10 text-amber-800 dark:text-amber-200'
                    : 'border-border/70 text-muted-foreground',
                )}
              >
                {observation.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => setStoppedEarly((value) => !value)}
            className={cn(
              'mt-4 flex w-full items-center justify-between rounded-xl border px-3 py-2 text-left text-xs',
              stoppedEarly
                ? 'border-amber-500/40 bg-amber-500/10'
                : 'border-border/70 bg-background/50',
            )}
          >
            <span>We stopped early</span>
            {stoppedEarly && <Check className="h-4 w-4" aria-hidden="true" />}
          </button>
        </details>

        <Button
          size="lg"
          className="mt-7 h-12 w-full rounded-2xl"
          disabled={attempts === 0 || saveMutation.isPending}
          onClick={() => saveMutation.mutate()}
        >
          {saveMutation.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <Heart className="mr-2 h-4 w-4" aria-hidden="true" />
          )}
          Save practice
        </Button>
        {attempts === 0 && (
          <p className="mt-2 text-center text-xs text-muted-foreground">
            Log at least one observable repetition before saving.
          </p>
        )}
      </div>
    </div>
  );
}

export default function CoachPage() {
  const queryClient = useQueryClient();
  const [practicePlan, setPracticePlan] = useState<CoachingPlan | null>(null);
  const [latestDecision, setLatestDecision] = useState<CoachingProgression | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ['coaching', 'me'],
    queryFn: () => coachingApi.getMine(),
  });

  const startMutation = useMutation({
    mutationFn: ({ petId, templateId }: { petId: string; templateId: string }) =>
      coachingApi.startPlan(petId, templateId),
    onSuccess: () => {
      setLatestDecision(null);
      void queryClient.invalidateQueries({ queryKey: ['coaching', 'me'] });
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ planId, status }: { planId: string; status: 'ACTIVE' | 'PAUSED' }) =>
      coachingApi.setPlanStatus(planId, status),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ['coaching', 'me'] }),
  });

  const completion = useMemo(() => {
    if (!data?.activePlan?.recentSuccessRate) return null;
    return Math.round(data.activePlan.recentSuccessRate * 100);
  }, [data?.activePlan?.recentSuccessRate]);

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
              Practice, notice, adapt
            </p>
            <h1 className="text-xl font-bold tracking-tight">Coach</h1>
          </div>
          {data?.weeklyRhythm && (
            <div className="text-right text-xs text-muted-foreground">
              <p className="font-semibold text-foreground">{data.weeklyRhythm.sessions} sessions</p>
              <p>{data.weeklyRhythm.minutes} min this week</p>
            </div>
          )}
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        {isLoading ? (
          <div className="flex min-h-[55vh] items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
            <span className="sr-only">Loading coaching plan</span>
          </div>
        ) : error ? (
          <div className="rounded-3xl border border-border/60 bg-card p-6 text-center">
            <CircleAlert className="mx-auto h-6 w-6 text-muted-foreground" aria-hidden="true" />
            <h2 className="mt-3 font-semibold">Coach could not load</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Your training history is unchanged. Try again when the connection settles.
            </p>
          </div>
        ) : !data?.pet ? (
          <div className="rounded-3xl border border-border/60 bg-card p-6 text-center">
            <Brain className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
            <h2 className="mt-3 text-lg font-semibold">Add a pet to begin</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Coaching is built around an individual pet, their routine, and what you observe
              together.
            </p>
          </div>
        ) : data.activePlan ? (
          <ActivePlan
            dashboard={data}
            completion={completion}
            latestDecision={latestDecision}
            onPractice={() => setPracticePlan(data.activePlan)}
            onPause={() =>
              statusMutation.mutate({ planId: data.activePlan!.id, status: 'PAUSED' })
            }
          />
        ) : (
          <TemplatePicker
            dashboard={data}
            pending={startMutation.isPending}
            onStart={(templateId) =>
              startMutation.mutate({ petId: data.pet!.id, templateId })
            }
            onResume={(planId) => statusMutation.mutate({ planId, status: 'ACTIVE' })}
          />
        )}
      </main>

      <BottomNav />

      {practicePlan && (
        <PracticePanel
          plan={practicePlan}
          onClose={() => setPracticePlan(null)}
          onSaved={(decision) => {
            setPracticePlan(null);
            setLatestDecision(decision);
          }}
        />
      )}
    </div>
  );
}

function ActivePlan({
  dashboard,
  completion,
  latestDecision,
  onPractice,
  onPause,
}: {
  dashboard: CoachingDashboard;
  completion: number | null;
  latestDecision: CoachingProgression | null;
  onPractice: () => void;
  onPause: () => void;
}) {
  const plan = dashboard.activePlan!;
  return (
    <>
      <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.09] via-card/80 to-secondary/[0.06] p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow">{dashboard.pet!.name}&apos;s focus</p>
            <h2 className="mt-1 text-2xl font-bold tracking-tight">{plan.title}</h2>
            <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
              {plan.objective}
            </p>
          </div>
          <span className="shrink-0 rounded-full border border-primary/20 bg-background/70 px-3 py-1 text-xs font-semibold text-primary">
            Level {plan.level}
          </span>
        </div>

        <div className="mt-5 grid grid-cols-3 gap-2 text-center">
          <div className="rounded-2xl bg-background/60 px-2 py-3">
            <p className="text-lg font-bold">{plan.sessionsCompleted}</p>
            <p className="text-[10px] text-muted-foreground">sessions</p>
          </div>
          <div className="rounded-2xl bg-background/60 px-2 py-3">
            <p className="text-lg font-bold">{completion ?? '—'}{completion !== null ? '%' : ''}</p>
            <p className="text-[10px] text-muted-foreground">recent success</p>
          </div>
          <div className="rounded-2xl bg-background/60 px-2 py-3">
            <p className="text-lg font-bold">{dashboard.weeklyRhythm.minutes}</p>
            <p className="text-[10px] text-muted-foreground">minutes / week</p>
          </div>
        </div>

        <Button size="lg" className="mt-5 h-12 w-full rounded-2xl" onClick={onPractice}>
          <Play className="mr-2 h-4 w-4" aria-hidden="true" />
          Start a short practice
        </Button>
      </section>

      <section className="mt-4">
        <ProgressionCard decision={latestDecision ?? plan.nextPractice} />
      </section>

      <section className="mt-7">
        <p className="eyebrow">Set up the win</p>
        <h2 className="mt-1 text-xl font-bold tracking-tight">Today&apos;s practice</h2>
        <div className="mt-3 space-y-2">
          {plan.steps.map((step, index) => (
            <div key={step} className="surface-soft flex gap-3 rounded-2xl p-4">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                {index + 1}
              </span>
              <p className="pt-0.5 text-sm leading-relaxed">{step}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-6 grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-border/60 bg-card/65 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Heart className="h-4 w-4 text-primary" aria-hidden="true" />
            Find the reward
          </div>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            {plan.rewardExamples.join(' · ')}
          </p>
        </div>
        <div className="rounded-2xl border border-border/60 bg-card/65 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Timer className="h-4 w-4 text-primary" aria-hidden="true" />
            Keep it light
          </div>
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            Stop while the exercise is still easy enough to succeed. More repetitions are not
            automatically better.
          </p>
        </div>
      </section>

      <div className={cn('mt-5 rounded-2xl border p-4', plan.support.recommended ? 'border-amber-500/30 bg-amber-500/[0.07]' : 'border-border/60 bg-card/50')}>
        <p className="text-xs leading-relaxed text-muted-foreground">{plan.support.message}</p>
      </div>

      <details className="mt-5 rounded-2xl border border-border/60 bg-card/50 p-4">
        <summary className="cursor-pointer text-sm font-semibold">Why Coach works this way</summary>
        <ul className="mt-3 space-y-2 text-xs leading-relaxed text-muted-foreground">
          {dashboard.methodology.principles.map((principle) => (
            <li key={principle} className="flex gap-2">
              <span className="text-primary" aria-hidden="true">•</span>
              <span>{principle}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
          {dashboard.methodology.progressionPolicy}
        </p>
      </details>

      <Button variant="ghost" className="mt-4 w-full text-muted-foreground" onClick={onPause}>
        <Pause className="mr-2 h-4 w-4" aria-hidden="true" />
        Pause this focus
      </Button>
    </>
  );
}

function TemplatePicker({
  dashboard,
  pending,
  onStart,
  onResume,
}: {
  dashboard: CoachingDashboard;
  pending: boolean;
  onStart: (templateId: string) => void;
  onResume: (planId: string) => void;
}) {
  return (
    <>
      <section className="rounded-3xl border border-border/60 bg-card/70 p-5">
        <p className="eyebrow">One focus at a time</p>
        <h2 className="mt-1 text-2xl font-bold tracking-tight">
          What would make life easier with {dashboard.pet!.name}?
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Pick one everyday skill. Woof will keep the first repetitions easy, watch what you log,
          and change only one layer of difficulty at a time.
        </p>
      </section>

      {dashboard.pausedPlans.length > 0 && (
        <section className="mt-6">
          <p className="eyebrow">Continue</p>
          <div className="mt-2 space-y-2">
            {dashboard.pausedPlans.slice(0, 2).map((plan) => (
              <button
                key={plan.id}
                type="button"
                onClick={() => onResume(plan.id)}
                className="surface-soft flex w-full items-center justify-between rounded-2xl p-4 text-left"
              >
                <div>
                  <p className="font-semibold">{plan.title}</p>
                  <p className="mt-1 text-xs text-muted-foreground">Resume at level {plan.level}</p>
                </div>
                <ChevronRight className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="mt-7">
        <p className="eyebrow">Starter skills</p>
        <div className="mt-3 space-y-3">
          {dashboard.templates.map((template) => (
            <button
              key={template.id}
              type="button"
              disabled={pending}
              onClick={() => onStart(template.id)}
              className="group surface-soft flex w-full items-start gap-4 rounded-2xl p-4 text-left transition-colors hover:border-primary/30 hover:bg-primary/[0.04] disabled:opacity-60"
            >
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Brain className="h-5 w-5" aria-hidden="true" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-semibold">{template.title}</span>
                <span className="mt-1 block text-sm leading-relaxed text-muted-foreground">
                  {template.objective}
                </span>
                <span className="mt-2 block text-xs font-semibold text-primary">
                  Start easy <span aria-hidden="true">→</span>
                </span>
              </span>
            </button>
          ))}
        </div>
      </section>

      <div className="mt-6 rounded-2xl border border-border/60 bg-card/40 p-4">
        <p className="text-xs leading-relaxed text-muted-foreground">
          Coach is for everyday learning and cooperative skills. Persistent aggression, fear,
          panic, pain, or major behavior change deserves assessment from a veterinarian or a
          qualified reward-based behavior professional.
        </p>
      </div>
    </>
  );
}
