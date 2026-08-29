'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, CheckCircle2, Clock3, Loader2, LockKeyhole, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import {
  caregiverApi,
  type CaregiverObservationKind,
  type CaregiverToday,
} from '@/lib/api/caregiver';

const observationKinds: Array<{ value: CaregiverObservationKind; label: string }> = [
  { value: 'ROUTINE', label: 'Routine' },
  { value: 'ACTIVITY_RESPONSE', label: 'Activity response' },
  { value: 'BEHAVIOR', label: 'Behavior' },
  { value: 'HANDOFF_NOTE', label: 'Handoff note' },
];

function expiryCopy(expiresAt: string) {
  const date = new Date(expiresAt);
  if (Number.isNaN(date.getTime())) return 'Expiry unavailable';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function AuthorityUnavailable({ onRetry }: { onRetry?: () => void }) {
  return (
    <main id="main-content" className="mx-auto max-w-xl px-4 py-12">
      <section className="rounded-3xl border border-border/70 bg-card/60 p-6 text-center">
        <LockKeyhole className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
        <h1 className="mt-3 text-xl font-bold">Caregiver access is not available</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          Temporary access may have expired, been revoked, or become unavailable because the
          relationship changed. Woof keeps pet details closed when current authority cannot be
          proven.
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {onRetry && (
            <Button variant="outline" className="bg-transparent" onClick={onRetry}>
              Check again
            </Button>
          )}
          <Button asChild>
            <Link href="/">Back to Today</Link>
          </Button>
        </div>
      </section>
    </main>
  );
}

function BoundaryGrid({ data }: { data: CaregiverToday }) {
  const boundaries = [
    ['Household history', data.boundaries.householdHistory],
    ['Sibling pets', data.boundaries.siblingPets],
    ['Medical authority', data.boundaries.medicalAuthority],
    ['Profile correction', data.boundaries.profileCorrection],
    ['Connector administration', data.boundaries.connectorAdmin],
    ['Bond XP authority', data.boundaries.bondXpAuthority],
    ['Recommendation evidence authority', data.boundaries.recommendationEvidenceAuthority],
  ] as const;

  return (
    <section className="mt-6 rounded-3xl border border-border/70 bg-card/50 p-5">
      <p className="eyebrow">Deliberately not included</p>
      <h2 className="mt-1 text-lg font-bold">Temporary care stays narrow.</h2>
      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        {boundaries.map(([label, allowed]) => (
          <div
            key={label}
            className="flex items-center gap-2 rounded-2xl border border-border/60 bg-background/60 px-3 py-2.5 text-sm"
          >
            <LockKeyhole className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span>{label}</span>
            <span className="ml-auto text-xs font-semibold text-muted-foreground">
              {allowed ? 'Allowed' : 'Not granted'}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function CaregiverPetTodayPage() {
  const params = useParams<{ petId: string }>();
  const petId = params.petId;
  const queryClient = useQueryClient();
  const [clock, setClock] = useState(() => Date.now());
  const [kind, setKind] = useState<CaregiverObservationKind>('ROUTINE');
  const [summary, setSummary] = useState('');
  const [note, setNote] = useState('');

  const today = useQuery({
    queryKey: ['caregiver', 'today', petId],
    queryFn: () => caregiverApi.today(petId),
    retry: false,
    refetchInterval: 15_000,
    refetchOnWindowFocus: true,
  });

  const expiresAtMs = useMemo(
    () => (today.data ? new Date(today.data.relationship.expiresAt).getTime() : Number.NaN),
    [today.data]
  );

  useEffect(() => {
    if (!Number.isFinite(expiresAtMs)) return;
    const delay = Math.max(0, expiresAtMs - Date.now()) + 50;
    const timeout = window.setTimeout(() => {
      setClock(Date.now());
      void today.refetch();
    }, delay);
    return () => window.clearTimeout(timeout);
  }, [expiresAtMs, today]);

  const observation = useMutation({
    mutationFn: () =>
      caregiverApi.observe(petId, {
        kind,
        summary: summary.trim(),
        ...(note.trim() ? { note: note.trim() } : {}),
      }),
    onSuccess: async () => {
      setSummary('');
      setNote('');
      await queryClient.invalidateQueries({ queryKey: ['caregiver', 'today', petId] });
    },
    onError: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['caregiver', 'today', petId] }),
        queryClient.invalidateQueries({ queryKey: ['caregiver', 'active-pets'] }),
      ]);
    },
  });

  if (today.isLoading) {
    return (
      <main id="main-content" className="flex min-h-screen items-center justify-center" role="status">
        <div className="text-center">
          <Loader2 className="mx-auto h-7 w-7 animate-spin text-primary" aria-hidden="true" />
          <p className="mt-3 text-sm text-muted-foreground">Checking caregiver authority…</p>
        </div>
      </main>
    );
  }

  if (today.isError || !today.data) {
    return <AuthorityUnavailable onRetry={() => void today.refetch()} />;
  }

  const locallyExpired = Number.isFinite(expiresAtMs) && clock >= expiresAtMs;
  if (today.data.relationship.effectiveStatus !== 'ACTIVE' || locallyExpired) {
    return <AuthorityUnavailable onRetry={() => void today.refetch()} />;
  }

  const data = today.data;
  const canObserve = data.available.logObservation;

  return (
    <div className="min-h-screen">
      <header className="border-b border-border/60 bg-background/92 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <Button asChild variant="ghost" size="icon" aria-label="Back to Today">
            <Link href="/">
              <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Temporary caregiver
            </p>
            <h1 className="text-lg font-bold tracking-tight">Today with {data.pet.name}</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-10 pt-5">
        <section
          data-caregiver-today
          className="rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.1] via-card/95 to-secondary/[0.06] p-6"
        >
          <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <ShieldCheck className="h-5 w-5" aria-hidden="true" />
          </span>
          <p className="eyebrow mt-4">Current pet-scoped authority</p>
          <h2 className="mt-1 text-3xl font-bold tracking-tight">Care for {data.pet.name} today.</h2>
          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
            @{data.relationship.issuerHandle ?? 'guardian'} granted this temporary view. Opening or
            refreshing this page checks current server authority again.
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-border/70 bg-background/65 px-3 py-1.5 text-xs font-semibold text-muted-foreground">
            <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
            Access expires {expiryCopy(data.relationship.expiresAt)}
          </div>
        </section>

        <section className="mt-6 rounded-3xl border border-border/70 bg-card/60 p-5">
          <p className="eyebrow">What you can do</p>
          <h2 className="mt-1 text-lg font-bold">Present-tense care only.</h2>
          <div className="mt-4 space-y-2 text-sm text-muted-foreground">
            <p>• View this caregiver-safe Today context.</p>
            <p>
              • {canObserve ? 'Leave context-only care observations.' : 'Observation writing was not granted.'}
            </p>
          </div>
        </section>

        {canObserve && (
          <section className="mt-6 rounded-3xl border border-border/70 bg-card/60 p-5">
            <p className="eyebrow">Care observation</p>
            <h2 className="mt-1 text-lg font-bold">Leave useful context, not a permanent dog trait.</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              This note is stored as caregiver context only. It does not award Bond XP, complete an
              Adventure, update the dog's profile, or become recommendation evidence automatically.
            </p>

            <label className="mt-5 block text-sm font-semibold" htmlFor="caregiver-observation-kind">
              What kind of observation is this?
            </label>
            <select
              id="caregiver-observation-kind"
              data-caregiver-observation-kind
              className="mt-2 h-11 w-full rounded-xl border border-input bg-background px-3 text-sm"
              value={kind}
              onChange={(event) => setKind(event.target.value as CaregiverObservationKind)}
            >
              {observationKinds.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            <label className="mt-4 block text-sm font-semibold" htmlFor="caregiver-observation-summary">
              Short observation
            </label>
            <textarea
              id="caregiver-observation-summary"
              data-caregiver-observation-summary
              className="mt-2 min-h-24 w-full resize-y rounded-xl border border-input bg-background px-3 py-2 text-sm"
              maxLength={240}
              value={summary}
              onChange={(event) => setSummary(event.target.value)}
              placeholder="Example: Settled after the evening routine and chose to rest by the door."
            />

            <label className="mt-4 block text-sm font-semibold" htmlFor="caregiver-observation-note">
              Optional handoff detail
            </label>
            <textarea
              id="caregiver-observation-note"
              className="mt-2 min-h-20 w-full resize-y rounded-xl border border-input bg-background px-3 py-2 text-sm"
              maxLength={500}
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Anything the guardian may want to know later."
            />

            <Button
              data-caregiver-submit-observation
              className="mt-4"
              disabled={observation.isPending || summary.trim().length === 0}
              onClick={() => observation.mutate()}
            >
              {observation.isPending && (
                <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden="true" />
              )}
              Save context-only observation
            </Button>

            {observation.isSuccess && (
              <p
                data-caregiver-observation-saved
                className="mt-4 flex items-center gap-2 text-sm font-medium text-primary"
                role="status"
              >
                <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                Observation saved as context only.
              </p>
            )}
            {observation.isError && (
              <p className="mt-4 text-sm text-destructive" role="alert">
                Woof could not prove current caregiver authority for that write. Nothing was saved.
              </p>
            )}
          </section>
        )}

        <BoundaryGrid data={data} />
      </main>
    </div>
  );
}
