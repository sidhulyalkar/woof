'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity,
  BatteryLow,
  BellRing,
  CalendarClock,
  Check,
  Clock3,
  Loader2,
  PawPrint,
  Plus,
  Radio,
  ShieldCheck,
  Sparkles,
  Trash2,
} from 'lucide-react';
import Link from 'next/link';
import { FormEvent, useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import {
  autopilotApi,
  type AutopilotSignal,
  type CareReminderKind,
  type CreateCareReminderInput,
} from '@/lib/api/autopilot';
import { householdsApi } from '@/lib/api/households';

const reminderKinds: Array<{ value: CareReminderKind; label: string }> = [
  { value: 'VET_APPOINTMENT', label: 'Vet appointment' },
  { value: 'MEDICATION', label: 'Medication' },
  { value: 'GROOMING', label: 'Grooming' },
  { value: 'GENERAL_CARE', label: 'General care' },
];

const inputClass =
  'h-11 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none transition focus:border-primary/60 focus:ring-2 focus:ring-primary/10';

function formatDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Invalid date';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

function signalIcon(signal: AutopilotSignal) {
  return signal.signalType === 'TRACKER_BATTERY_LOW' ? BatteryLow : Activity;
}

export default function AutopilotPage() {
  const queryClient = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [petId, setPetId] = useState('');
  const [kind, setKind] = useState<CareReminderKind>('GENERAL_CARE');
  const [title, setTitle] = useState('');
  const [detail, setDetail] = useState('');
  const [dueAt, setDueAt] = useState('');
  const [repeatEveryDays, setRepeatEveryDays] = useState('');

  const dashboard = useQuery({
    queryKey: ['autopilot'],
    queryFn: () => autopilotApi.getDashboard(),
    retry: false,
  });

  const households = useQuery({
    queryKey: ['households', 'me'],
    queryFn: () => householdsApi.getMine(),
    retry: false,
  });

  const pets = useMemo(() => {
    const byId = new Map<
      string,
      { id: string; name: string; species: string; avatarUrl?: string | null }
    >();
    for (const household of households.data ?? []) {
      for (const membership of household.pets) {
        byId.set(membership.pet.id, membership.pet);
      }
    }
    return [...byId.values()];
  }, [households.data]);

  const petNames = useMemo(() => new Map(pets.map((pet) => [pet.id, pet.name] as const)), [pets]);

  const createReminder = useMutation({
    mutationFn: (input: CreateCareReminderInput) => autopilotApi.createReminder(input),
    onSuccess: async () => {
      setTitle('');
      setDetail('');
      setDueAt('');
      setRepeatEveryDays('');
      setShowForm(false);
      await queryClient.invalidateQueries({ queryKey: ['autopilot'] });
    },
  });

  const cancelReminder = useMutation({
    mutationFn: (id: string) => autopilotApi.cancelReminder(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['autopilot'] }),
  });

  const acknowledgeSignal = useMutation({
    mutationFn: (id: string) => autopilotApi.acknowledgeSignal(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['autopilot'] }),
  });

  const submitReminder = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const due = new Date(dueAt);
    if (!title.trim() || !Number.isFinite(due.getTime())) return;

    const repeat = repeatEveryDays ? Number(repeatEveryDays) : undefined;
    createReminder.mutate({
      ...(petId ? { petId } : {}),
      kind,
      title: title.trim(),
      ...(detail.trim() ? { detail: detail.trim() } : {}),
      dueAt: due.toISOString(),
      ...(repeat && Number.isInteger(repeat) ? { repeatEveryDays: repeat } : {}),
    });
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <Link href="/" className="flex items-center gap-3" aria-label="Back to Woof Today">
            <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
              <Sparkles className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
            </span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                dogOS
              </span>
              <span className="block text-lg font-bold tracking-tight">Autopilot</span>
            </span>
          </Link>
          <Button size="sm" onClick={() => setShowForm((value) => !value)}>
            <Plus className="mr-1.5 h-4 w-4" aria-hidden="true" />
            Reminder
          </Button>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.1] via-card/90 to-secondary/[0.07] p-5 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <BellRing className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">Quietly proactive</p>
              <h1 className="mt-1 text-2xl font-bold tracking-tight">
                Useful context, without surrendering the wheel.
              </h1>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Autopilot keeps care reminders together and can surface conservative tracker
                check-ins. It does not diagnose your dog, award wearable XP, or store tracker GPS in
                this release.
              </p>
            </div>
          </div>
        </section>

        {showForm && (
          <section className="surface-soft mt-4 rounded-3xl p-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="eyebrow">Schedule care</p>
                <h2 className="mt-1 text-lg font-bold">New reminder</h2>
              </div>
              <Clock3 className="h-5 w-5 text-primary" aria-hidden="true" />
            </div>

            <form className="mt-4 space-y-4" onSubmit={submitReminder}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <label className="space-y-1.5 text-xs font-semibold text-muted-foreground">
                  Dog
                  <select
                    className={inputClass}
                    value={petId}
                    onChange={(e) => setPetId(e.target.value)}
                  >
                    <option value="">Household / general</option>
                    {pets.map((pet) => (
                      <option key={pet.id} value={pet.id}>
                        {pet.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1.5 text-xs font-semibold text-muted-foreground">
                  Type
                  <select
                    className={inputClass}
                    value={kind}
                    onChange={(e) => setKind(e.target.value as CareReminderKind)}
                  >
                    {reminderKinds.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <label className="block space-y-1.5 text-xs font-semibold text-muted-foreground">
                Reminder
                <input
                  className={inputClass}
                  value={title}
                  maxLength={120}
                  required
                  placeholder="e.g. Monthly heartworm medication"
                  onChange={(e) => setTitle(e.target.value)}
                />
              </label>

              <label className="block space-y-1.5 text-xs font-semibold text-muted-foreground">
                Note <span className="font-normal">optional</span>
                <input
                  className={inputClass}
                  value={detail}
                  maxLength={500}
                  placeholder="Context for your household"
                  onChange={(e) => setDetail(e.target.value)}
                />
              </label>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <label className="space-y-1.5 text-xs font-semibold text-muted-foreground">
                  Due
                  <input
                    className={inputClass}
                    type="datetime-local"
                    required
                    value={dueAt}
                    onChange={(e) => setDueAt(e.target.value)}
                  />
                </label>
                <label className="space-y-1.5 text-xs font-semibold text-muted-foreground">
                  Repeat every <span className="font-normal">days, optional</span>
                  <input
                    className={inputClass}
                    type="number"
                    min={1}
                    max={365}
                    inputMode="numeric"
                    value={repeatEveryDays}
                    placeholder="30"
                    onChange={(e) => setRepeatEveryDays(e.target.value)}
                  />
                </label>
              </div>

              {kind === 'MEDICATION' && (
                <p className="rounded-2xl border border-border/70 bg-background/55 p-3 text-xs leading-relaxed text-muted-foreground">
                  dogOS schedules the reminder only. Follow the dose and timing instructions from
                  your veterinarian or medication label.
                </p>
              )}

              {createReminder.isError && (
                <p className="text-sm text-destructive" role="alert">
                  That reminder could not be saved. Check the details and try again.
                </p>
              )}

              <div className="flex gap-2">
                <Button
                  type="submit"
                  disabled={createReminder.isPending || !title.trim() || !dueAt}
                >
                  {createReminder.isPending && (
                    <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden="true" />
                  )}
                  Save reminder
                </Button>
                <Button type="button" variant="ghost" onClick={() => setShowForm(false)}>
                  Cancel
                </Button>
              </div>
            </form>
          </section>
        )}

        {dashboard.isLoading ? (
          <div className="flex min-h-[35vh] items-center justify-center" role="status">
            <div className="text-center">
              <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
              <p className="mt-3 text-sm text-muted-foreground">Checking the household rhythm…</p>
            </div>
          </div>
        ) : dashboard.isError || !dashboard.data ? (
          <section className="surface-soft mt-4 rounded-3xl p-6 text-center">
            <ShieldCheck className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h2 className="mt-3 text-lg font-bold">Autopilot is unavailable</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              The feature may be paused for this environment. Your existing care, Adventure, and
              household records are unchanged.
            </p>
          </section>
        ) : (
          <>
            <section className="mt-6">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Needs your eyes</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">Check-ins</h2>
                </div>
                <span className="text-xs font-semibold text-muted-foreground">
                  {dashboard.data.signals.length} open
                </span>
              </div>

              <div className="mt-3 space-y-3">
                {dashboard.data.signals.length === 0 ? (
                  <div className="surface-soft rounded-2xl p-4 text-sm text-muted-foreground">
                    Nothing needs a check-in right now.
                  </div>
                ) : (
                  dashboard.data.signals.map((signal) => {
                    const Icon = signalIcon(signal);
                    return (
                      <article key={signal.id} className="surface-soft rounded-2xl p-4">
                        <div className="flex items-start gap-3">
                          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                            <Icon className="h-4 w-4" aria-hidden="true" />
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <h3 className="font-semibold">{signal.title}</h3>
                              <span className="rounded-full border border-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                                {signal.level === 'CHECK_IN' ? 'Check in' : 'Info'}
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-muted-foreground">
                              {petNames.get(signal.petId) ?? 'Your dog'} ·{' '}
                              {formatDate(signal.observedAt)}
                            </p>
                            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                              {signal.body}
                            </p>
                            <Button
                              className="mt-3"
                              variant="outline"
                              size="sm"
                              disabled={acknowledgeSignal.isPending}
                              onClick={() => acknowledgeSignal.mutate(signal.id)}
                            >
                              <Check className="mr-1.5 h-4 w-4" aria-hidden="true" />
                              Got it
                            </Button>
                          </div>
                        </div>
                      </article>
                    );
                  })
                )}
              </div>
            </section>

            <section className="mt-7">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Coming up</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">Care reminders</h2>
                </div>
                <CalendarClock className="h-5 w-5 text-primary" aria-hidden="true" />
              </div>

              <div className="mt-3 space-y-3">
                {dashboard.data.reminders.length === 0 ? (
                  <button
                    type="button"
                    className="surface-soft w-full rounded-2xl p-4 text-left text-sm text-muted-foreground transition hover:border-primary/30"
                    onClick={() => setShowForm(true)}
                  >
                    No reminders yet. Add one for appointments, medication, grooming, or ordinary
                    care.
                  </button>
                ) : (
                  dashboard.data.reminders.map((reminder) => (
                    <article key={reminder.id} className="surface-soft rounded-2xl p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-primary">
                          <BellRing className="h-4 w-4" aria-hidden="true" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <h3 className="font-semibold">{reminder.title}</h3>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {reminder.petId
                                  ? (petNames.get(reminder.petId) ?? 'Dog')
                                  : 'Household'}{' '}
                                · {formatDate(reminder.dueAt)}
                              </p>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              aria-label={`Cancel ${reminder.title}`}
                              disabled={cancelReminder.isPending}
                              onClick={() => cancelReminder.mutate(reminder.id)}
                            >
                              <Trash2 className="h-4 w-4" aria-hidden="true" />
                            </Button>
                          </div>
                          {reminder.detail && (
                            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                              {reminder.detail}
                            </p>
                          )}
                          {reminder.repeatEveryDays && (
                            <p className="mt-2 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                              Repeats every {reminder.repeatEveryDays} days
                            </p>
                          )}
                        </div>
                      </div>
                    </article>
                  ))
                )}
              </div>
            </section>

            <section className="mt-7">
              <p className="eyebrow">Tracker adapters</p>
              <h2 className="mt-1 text-xl font-bold tracking-tight">Ready for a real connector</h2>
              <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                {dashboard.data.providers.map((provider) => (
                  <article key={provider.provider} className="surface-soft rounded-2xl p-4">
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
                        <Radio className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div>
                        <h3 className="font-semibold">{provider.provider}</h3>
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                          Adapter ready
                        </p>
                      </div>
                    </div>
                    <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
                      Daily activity and device-health summaries are normalized. OAuth, webhooks,
                      revocation, and location permissions arrive in Connectors.
                    </p>
                  </article>
                ))}
              </div>
            </section>

            <section className="mt-7 rounded-3xl border border-primary/15 bg-primary/[0.04] p-5">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-5 w-5 text-primary" aria-hidden="true" />
                <h2 className="font-bold">Autopilot boundaries</h2>
              </div>
              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Boundary label="Tracker GPS stored" value="No" />
                <Boundary label="Wearables award Bond XP" value="No" />
                <Boundary label="Provider can edit dog records" value="No" />
                <Boundary label="Signals are diagnoses" value="No" />
              </div>
            </section>
          </>
        )}
      </main>

      <BottomNav />
    </div>
  );
}

function Boundary({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 bg-background/55 px-3 py-2.5">
      <span className="flex items-center gap-2 text-xs text-muted-foreground">
        <PawPrint className="h-3.5 w-3.5 text-primary" aria-hidden="true" />
        {label}
      </span>
      <span className="text-xs font-bold text-primary">{value}</span>
    </div>
  );
}
