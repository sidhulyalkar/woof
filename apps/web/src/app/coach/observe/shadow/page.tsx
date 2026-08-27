'use client';

import { useQuery } from '@tanstack/react-query';
import {
  ArrowLeft,
  CheckCircle2,
  CircleDashed,
  Clock3,
  FlaskConical,
  Loader2,
  LockKeyhole,
  ShieldCheck,
} from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { behaviorVisionApi } from '@/lib/api/behavior-vision';
import { useSessionStore } from '@/store/session';

function percent(value: number | null) {
  return value === null ? '—' : `${Math.round(value * 100)}%`;
}

function clipTime(milliseconds: number) {
  const seconds = Math.max(0, Math.round(milliseconds / 100) / 10);
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${minutes}:${remainder.toFixed(1).padStart(4, '0')}`;
}

export default function BehaviorShadowPage() {
  const user = useSessionStore((state) => state.user);
  const pets = useMemo(() => user?.pets ?? [], [user?.pets]);
  const [petId, setPetId] = useState(pets[0]?.id ?? '');

  useEffect(() => {
    if (!petId && pets[0]) setPetId(pets[0].id);
  }, [petId, pets]);

  const snapshot = useQuery({
    queryKey: ['behavior-shadow', petId],
    queryFn: () => behaviorVisionApi.shadow(petId),
    enabled: Boolean(petId),
  });

  const evaluation = snapshot.data?.evaluation;
  const gates = evaluation?.readinessGates;
  const gateRows =
    evaluation && gates
      ? [
          {
            label: 'Usable observations',
            value: `${evaluation.usableObservations} / ${gates.usableObservations}`,
            pass: evaluation.usableObservations >= gates.usableObservations,
          },
          {
            label: 'Owner-reviewed observations',
            value: `${evaluation.ownerReviewedObservations} / ${gates.ownerReviewedObservations}`,
            pass: evaluation.ownerReviewedObservations >= gates.ownerReviewedObservations,
          },
          {
            label: 'Owner confirmation',
            value: `${percent(evaluation.confirmationRate)} / ${Math.round(gates.confirmationRate * 100)}%`,
            pass:
              evaluation.confirmationRate !== null &&
              evaluation.confirmationRate >= gates.confirmationRate,
          },
          {
            label: 'Contexts sampled',
            value: `${evaluation.contextsSeen} / ${gates.contexts}`,
            pass: evaluation.contextsSeen >= gates.contexts,
          },
          {
            label: 'Paired sessions',
            value: `${evaluation.pairedSessions} / ${gates.pairedSessions}`,
            pass: evaluation.pairedSessions >= gates.pairedSessions,
          },
        ]
      : [];

  return (
    <div className="min-h-screen bg-background pb-28">
      <header className="sticky top-0 z-20 border-b border-border/50 bg-background/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-xl items-center gap-3 px-4 py-4">
          <Button asChild variant="ghost" size="icon">
            <Link href="/coach/observe" aria-label="Back to behavior observation">
              <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div className="min-w-0 flex-1">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
              Behavior Moments
            </p>
            <h1 className="truncate text-lg font-bold">Shadow Lab</h1>
          </div>
          <span className="rounded-full border border-border/60 bg-card px-3 py-1 text-xs font-semibold text-muted-foreground">
            zero authority
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-xl space-y-5 px-4 py-5">
        <section className="rounded-3xl border border-primary/15 bg-primary/[0.055] p-5">
          <div className="flex gap-3">
            <FlaskConical className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-semibold">Evidence can accumulate without becoming truth</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                This lab measures whether video observations are repeatable and owner-confirmed.
                Nothing here changes compatibility, canonical pet state, or safety decisions.
              </p>
            </div>
          </div>
        </section>

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

        {!petId ? (
          <section className="rounded-3xl border border-border/60 bg-card p-6 text-center">
            <p className="font-semibold">Add a pet before collecting Behavior Moments.</p>
          </section>
        ) : snapshot.isLoading ? (
          <div className="flex min-h-56 items-center justify-center gap-2" role="status">
            <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Loading shadow evidence…</span>
          </div>
        ) : snapshot.isError || !snapshot.data ? (
          <section className="rounded-3xl border border-border/60 bg-card p-6 text-center">
            <p className="font-semibold">Shadow evidence is temporarily unavailable.</p>
            <Button variant="outline" className="mt-4" onClick={() => snapshot.refetch()}>
              Try again
            </Button>
          </section>
        ) : (
          <>
            <section className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-border/60 bg-card p-4">
                <p className="text-xs text-muted-foreground">Usable evidence</p>
                <p className="mt-1 text-2xl font-bold">{evaluation?.usableObservations ?? 0}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {percent(evaluation?.usableRate ?? 0)} of active-release observations
                </p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-card p-4">
                <p className="text-xs text-muted-foreground">Owner confirmation</p>
                <p className="mt-1 text-2xl font-bold">
                  {percent(evaluation?.confirmationRate ?? null)}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {evaluation?.ownerReviewedObservations ?? 0} reviewed
                </p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-card p-4">
                <p className="text-xs text-muted-foreground">Contexts</p>
                <p className="mt-1 text-2xl font-bold">{evaluation?.contextsSeen ?? 0}</p>
                <p className="mt-1 text-xs text-muted-foreground">breadth matters</p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-card p-4">
                <p className="text-xs text-muted-foreground">Paired sessions</p>
                <p className="mt-1 text-2xl font-bold">{evaluation?.pairedSessions ?? 0}</p>
                <p className="mt-1 text-xs text-muted-foreground">baseline + change/recovery</p>
              </div>
            </section>

            <section className="rounded-3xl border border-border/60 bg-card/70 p-5">
              <div className="flex gap-3">
                <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <p className="eyebrow">Model evidence authority</p>
                  <h2 className="mt-2 font-semibold">Active qualified release only</h2>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    Personal baselines and research-readiness gates use only observations from the
                    exact model release active today. Older qualified releases remain visible for
                    audit, but are not mixed into the current measurement scale.
                  </p>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-3 gap-2">
                <div className="rounded-xl bg-muted/35 px-3 py-3">
                  <p className="text-[11px] text-muted-foreground">Active release</p>
                  <p className="mt-1 text-lg font-bold">
                    {evaluation?.activeReleaseObservations ?? 0}
                  </p>
                </div>
                <div className="rounded-xl bg-muted/35 px-3 py-3">
                  <p className="text-[11px] text-muted-foreground">Older qualified</p>
                  <p className="mt-1 text-lg font-bold">
                    {evaluation?.inactiveQualifiedObservations ?? 0}
                  </p>
                </div>
                <div className="rounded-xl bg-muted/35 px-3 py-3">
                  <p className="text-[11px] text-muted-foreground">Legacy unknown</p>
                  <p className="mt-1 text-lg font-bold">
                    {evaluation?.unqualifiedObservations ?? 0}
                  </p>
                </div>
              </div>
              <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
                {evaluation?.inactiveQualifiedObservations ?? 0} older qualified and{' '}
                {evaluation?.unqualifiedObservations ?? 0} legacy unqualified observations are
                excluded from current learning.
              </p>
              <p className="mt-2 break-words text-xs text-muted-foreground">
                Active release: {evaluation?.activeReleaseId ?? 'none configured'}
              </p>
            </section>

            <section className="rounded-3xl border border-border/60 bg-card/70 p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="eyebrow">Research readiness</p>
                  <h2 className="mt-2 font-semibold">
                    {evaluation?.evidenceReady ? 'Evidence gates met' : 'Still collecting evidence'}
                  </h2>
                </div>
                <span className="rounded-full bg-muted px-3 py-1 text-xs font-semibold text-muted-foreground">
                  {evaluation?.evidenceReady ? 'ready to evaluate' : 'shadow only'}
                </span>
              </div>
              <div className="mt-4 space-y-2">
                {gateRows.map((gate) => (
                  <div
                    key={gate.label}
                    className="flex items-center justify-between gap-3 rounded-xl border border-border/50 px-3 py-2.5"
                  >
                    <span className="flex items-center gap-2 text-sm">
                      {gate.pass ? (
                        <CheckCircle2 className="h-4 w-4 text-primary" aria-hidden="true" />
                      ) : (
                        <CircleDashed
                          className="h-4 w-4 text-muted-foreground"
                          aria-hidden="true"
                        />
                      )}
                      {gate.label}
                    </span>
                    <span className="text-xs font-semibold text-muted-foreground">
                      {gate.value}
                    </span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
                Passing every gate only makes the active-release evidence substantial enough to
                evaluate. It does not switch on production authority.
              </p>
            </section>

            <section className="rounded-3xl border border-border/60 bg-card/70 p-5">
              <div className="flex items-center gap-2">
                <Clock3 className="h-4 w-4 text-primary" aria-hidden="true" />
                <h2 className="font-semibold">Reviewable moments</h2>
              </div>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                Timestamps point to evidence windows from transient clips. Raw video is not retained
                by this timeline.
              </p>
              {snapshot.data.moments.length === 0 ? (
                <p className="mt-4 rounded-xl bg-muted/40 p-4 text-sm text-muted-foreground">
                  No timestamped active-release video evidence yet. New model runs can populate
                  moments when the model returns timed evidence.
                </p>
              ) : (
                <div className="mt-4 space-y-3">
                  {snapshot.data.moments.slice(0, 12).map((moment, index) => (
                    <div
                      key={`${moment.observationId}-${moment.startMs}-${index}`}
                      className="rounded-2xl border border-border/60 p-4"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="text-xs font-semibold text-primary">
                          {clipTime(moment.startMs)}–{clipTime(moment.endMs)}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {Math.round(moment.confidence * 100)}% evidence confidence
                        </span>
                      </div>
                      <p className="mt-2 text-sm font-medium">{moment.labels.join(' · ')}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Sources: {moment.sources.join(', ')}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-3xl border border-border/60 bg-muted/25 p-5">
              <div className="flex gap-3">
                <LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div>
                  <p className="text-sm font-semibold">Authority remains locked</p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    Compatibility influence: off. Canonical pet mutation: off. Safety-decision
                    authority: off. Promotion needs a separate qualified release even if every
                    research gate passes.
                  </p>
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
                <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                Policy {snapshot.data.policy.version}
              </div>
            </section>
          </>
        )}
      </main>

      <BottomNav />
    </div>
  );
}
