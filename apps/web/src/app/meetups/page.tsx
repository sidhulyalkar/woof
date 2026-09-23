'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { CalendarDays, Check, Loader2, MapPin, MessageCircle, ShieldCheck, X } from 'lucide-react';
import Link from 'next/link';
import { useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { authApi } from '@/lib/api';
import {
  meetupProposalsApi,
  type MeetupOutcome,
  type MeetupParticipantOutcome,
  type MeetupProposal,
} from '@/lib/api/meetup-proposals';
import { useAuthStore } from '@/lib/stores/auth-store';

function readableDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Time to be confirmed';
  return new Intl.DateTimeFormat(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date);
}

function venueName(proposal: MeetupProposal) {
  return proposal.suggestedVenue?.name?.trim() || 'Public meetup place';
}

function ChoiceGroup<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T | null;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
}) {
  return (
    <div>
      <p className="text-xs font-semibold text-muted-foreground">{label}</p>
      <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`rounded-full border px-3 py-2 text-sm font-semibold transition-colors ${
              value === option.value
                ? 'border-primary/30 bg-primary/10 text-primary'
                : 'border-border/70 bg-background/60 text-muted-foreground hover:text-foreground'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

const dogExperienceCopy: Record<NonNullable<MeetupOutcome['dogExperience']>, string> = {
  loved_it: 'Loved it',
  comfortable: 'Comfortable',
  not_their_thing: 'Not for them',
};

const ownerExperienceCopy: Record<NonNullable<MeetupOutcome['ownerExperience']>, string> = {
  great: 'Easy',
  fine: 'Fine',
  a_lot_today: 'A lot today',
};

const meetAgainCopy: Record<NonNullable<MeetupOutcome['meetAgain']>, string> = {
  yes: 'Yes',
  maybe: 'Maybe',
  no: 'No',
};

function OutcomeCard({
  proposal,
  existingOutcome,
}: {
  proposal: MeetupProposal;
  existingOutcome: MeetupParticipantOutcome | null;
}) {
  const queryClient = useQueryClient();
  const [dogExperience, setDogExperience] = useState<NonNullable<
    MeetupOutcome['dogExperience']
  > | null>(null);
  const [ownerExperience, setOwnerExperience] = useState<NonNullable<
    MeetupOutcome['ownerExperience']
  > | null>(null);
  const [meetAgain, setMeetAgain] = useState<NonNullable<MeetupOutcome['meetAgain']> | null>(null);
  const [safe, setSafe] = useState<boolean | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: (outcome: MeetupOutcome) => meetupProposalsApi.complete(proposal.id, outcome),
    onSuccess: async (result) => {
      setMessage(
        result.reportSuggested
          ? 'Your private reflection was saved. You flagged a safety concern, so reporting or blocking may be useful.'
          : 'Your private reflection was saved. Woof can use your answer without exposing it to the other participant.'
      );
      await queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
    },
    onError: () => {
      setMessage('That reflection could not be saved. Refresh before trying a different answer.');
    },
  });

  const complete = () => {
    if (!dogExperience || !ownerExperience || !meetAgain || safe === null) return;
    submit.mutate({
      occurred: true,
      dogExperience,
      ownerExperience,
      meetAgain,
      checklistOk: safe,
    });
  };

  if (existingOutcome) {
    const repeatPlanningEligible =
      existingOutcome.occurred &&
      (existingOutcome.meetAgain === 'yes' || existingOutcome.meetAgain === 'maybe');

    return (
      <div className="mt-4 space-y-4 border-t border-border/60 pt-4">
        <div>
          <p className="eyebrow">Your private reflection</p>
          <h3 className="mt-1 font-semibold">
            {existingOutcome.occurred ? 'You closed the loop.' : 'You marked this as not happening.'}
          </h3>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            This is your answer only. Woof does not reveal the other participant&apos;s reflection or
            imply that they answered the same way.
          </p>
        </div>

        {existingOutcome.occurred ? (
          <dl className="grid gap-2 rounded-xl bg-muted/40 p-3 text-sm sm:grid-cols-2">
            {existingOutcome.dogExperience && (
              <div>
                <dt className="text-xs text-muted-foreground">For your dog</dt>
                <dd className="font-semibold">{dogExperienceCopy[existingOutcome.dogExperience]}</dd>
              </div>
            )}
            {existingOutcome.ownerExperience && (
              <div>
                <dt className="text-xs text-muted-foreground">For you</dt>
                <dd className="font-semibold">
                  {ownerExperienceCopy[existingOutcome.ownerExperience]}
                </dd>
              </div>
            )}
            {existingOutcome.meetAgain && (
              <div>
                <dt className="text-xs text-muted-foreground">Your meet-again answer</dt>
                <dd className="font-semibold">{meetAgainCopy[existingOutcome.meetAgain]}</dd>
              </div>
            )}
            {existingOutcome.checklistOk !== null && (
              <div>
                <dt className="text-xs text-muted-foreground">Felt safe</dt>
                <dd className="font-semibold">{existingOutcome.checklistOk ? 'Yes' : 'No'}</dd>
              </div>
            )}
          </dl>
        ) : (
          <p className="rounded-xl bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
            One private reflection does not automatically cancel the shared coordination record.
          </p>
        )}

        {existingOutcome.checklistOk === false && (
          <p className="rounded-xl border border-destructive/20 bg-destructive/5 p-3 text-xs leading-relaxed text-destructive">
            You flagged a safety concern. Consider using the reporting or blocking tools if that
            would help you feel safer.
          </p>
        )}

        {repeatPlanningEligible && (
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-3">
            <p className="text-sm font-semibold">Want another shared activity?</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Your answer says another meetup may be worthwhile. Open the conversation and suggest a
              new public-place plan when you want to. This does not reveal their answer.
            </p>
            <Button asChild size="sm" className="mt-3">
              <Link href="/inbox">
                <MessageCircle className="mr-2 h-4 w-4" aria-hidden="true" />
                Open messages
              </Link>
            </Button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4 border-t border-border/60 pt-4">
      <div>
        <p className="eyebrow">Close the loop</p>
        <h3 className="mt-1 font-semibold">How did it go?</h3>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          Three quick answers become your private relationship evidence. They are not a health
          diagnosis and are not shown to the other participant.
        </p>
      </div>

      <ChoiceGroup
        label="How was it for your dog?"
        value={dogExperience}
        onChange={setDogExperience}
        options={[
          { value: 'loved_it', label: 'Loved it' },
          { value: 'comfortable', label: 'Comfortable' },
          { value: 'not_their_thing', label: 'Not for them' },
        ]}
      />
      <ChoiceGroup
        label="How was it for you?"
        value={ownerExperience}
        onChange={setOwnerExperience}
        options={[
          { value: 'great', label: 'Easy' },
          { value: 'fine', label: 'Fine' },
          { value: 'a_lot_today', label: 'A lot today' },
        ]}
      />
      <ChoiceGroup
        label="Meet again?"
        value={meetAgain}
        onChange={setMeetAgain}
        options={[
          { value: 'yes', label: 'Yes' },
          { value: 'maybe', label: 'Maybe' },
          { value: 'no', label: 'No' },
        ]}
      />

      <div>
        <p className="text-xs font-semibold text-muted-foreground">Did the meetup feel safe?</p>
        <div className="mt-2 flex gap-2" role="group" aria-label="Did the meetup feel safe?">
          <Button
            type="button"
            size="sm"
            variant={safe === true ? 'default' : 'outline'}
            className={safe === true ? '' : 'bg-transparent'}
            onClick={() => setSafe(true)}
          >
            <ShieldCheck className="mr-2 h-4 w-4" aria-hidden="true" />
            Yes
          </Button>
          <Button
            type="button"
            size="sm"
            variant={safe === false ? 'destructive' : 'outline'}
            className={safe === false ? '' : 'bg-transparent'}
            onClick={() => setSafe(false)}
          >
            No
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          disabled={
            submit.isPending || !dogExperience || !ownerExperience || !meetAgain || safe === null
          }
          onClick={complete}
        >
          {submit.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />}
          Save outcome
        </Button>
        {proposal.status === 'accepted' && (
          <Button
            variant="ghost"
            disabled={submit.isPending}
            onClick={() => submit.mutate({ occurred: false })}
          >
            It didn&apos;t happen
          </Button>
        )}
      </div>
      {message && (
        <p className="rounded-xl bg-muted/50 p-3 text-xs text-muted-foreground" role="status">
          {message}
        </p>
      )}
    </div>
  );
}

export default function MeetupsPage() {
  const cachedUser = useAuthStore((state) => state.user);
  const queryClient = useQueryClient();
  const profile = useQuery({
    queryKey: ['auth-profile'],
    queryFn: authApi.me,
    staleTime: 30_000,
    retry: false,
  });
  const user = profile.data ?? cachedUser;
  const proposals = useQuery({
    queryKey: ['meetup-proposals'],
    queryFn: meetupProposalsApi.getMine,
    enabled: Boolean(user),
    retry: false,
  });

  const status = useMutation({
    mutationFn: ({ id, next }: { id: string; next: 'accepted' | 'declined' }) =>
      meetupProposalsApi.updateStatus(id, next),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['meetup-proposals'] });
    },
  });

  const items = useMemo(() => {
    if (!proposals.data || !user) return [];
    return [
      ...proposals.data.received.map((proposal) => ({ proposal, direction: 'received' as const })),
      ...proposals.data.sent.map((proposal) => ({ proposal, direction: 'sent' as const })),
    ].sort(
      (a, b) =>
        new Date(b.proposal.suggestedTime).getTime() - new Date(a.proposal.suggestedTime).getTime()
    );
  }, [proposals.data, user]);

  const outcomeByProposalId = useMemo(
    () =>
      new Map(
        (proposals.data?.outcomes ?? []).map((outcome) => [outcome.proposalId, outcome] as const)
      ),
    [proposals.data?.outcomes]
  );

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <div>
            <p className="eyebrow">Real-world loop</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Meetups</h1>
          </div>
          <Button asChild size="sm" variant="outline" className="bg-transparent">
            <Link href="/inbox">
              <MessageCircle className="mr-2 h-4 w-4" aria-hidden="true" />
              Messages
            </Link>
          </Button>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl space-y-4 px-4 py-5">
        <Card className="surface-soft rounded-2xl p-4">
          <p className="text-sm font-semibold">Keep coordination simple and public-place first</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Woof meetup proposals store a venue label and area, not a private home coordinate.
            Blocking either participant disables coordination.
          </p>
        </Card>

        {!user || proposals.isLoading ? (
          <div className="flex min-h-64 items-center justify-center gap-3" role="status">
            <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
            <span className="text-sm text-muted-foreground">Loading meetups…</span>
          </div>
        ) : proposals.isError ? (
          <Card className="surface-soft rounded-2xl p-6 text-center">
            <h2 className="font-semibold">Meetups are temporarily unavailable</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              No substitute plans are shown while canonical meetup records cannot be read.
            </p>
            <Button
              variant="outline"
              className="mt-4 bg-transparent"
              onClick={() => proposals.refetch()}
            >
              Try again
            </Button>
          </Card>
        ) : items.length === 0 ? (
          <Card className="surface-soft rounded-2xl p-6 text-center">
            <CalendarDays className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
            <h2 className="mt-3 font-semibold">No meetup plans yet</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Start with an explainable match, chat first, then suggest a public place when it feels
              right.
            </p>
            <Button asChild className="mt-5">
              <Link href="/discover">Find a compatible dog</Link>
            </Button>
          </Card>
        ) : (
          items.map(({ proposal, direction }) => (
            <Card key={`${direction}-${proposal.id}`} className="rounded-2xl p-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="eyebrow">
                    {direction === 'received' ? 'Request for you' : 'You suggested'}
                  </p>
                  <h2 className="mt-1 text-lg font-bold">{venueName(proposal)}</h2>
                  <div className="mt-2 flex flex-col gap-1 text-sm text-muted-foreground">
                    <span className="flex items-center gap-2">
                      <CalendarDays className="h-4 w-4" aria-hidden="true" />
                      {readableDate(proposal.suggestedTime)}
                    </span>
                    {proposal.suggestedVenue.area && (
                      <span className="flex items-center gap-2">
                        <MapPin className="h-4 w-4" aria-hidden="true" />
                        {proposal.suggestedVenue.area}
                      </span>
                    )}
                  </div>
                </div>
                <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-semibold capitalize text-muted-foreground">
                  {proposal.status}
                </span>
              </div>

              {proposal.status === 'pending' && direction === 'received' && (
                <div className="mt-4 flex gap-2 border-t border-border/60 pt-4">
                  <Button
                    size="sm"
                    disabled={status.isPending}
                    onClick={() => status.mutate({ id: proposal.id, next: 'accepted' })}
                  >
                    <Check className="mr-2 h-4 w-4" aria-hidden="true" />
                    Accept
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-transparent"
                    disabled={status.isPending}
                    onClick={() => status.mutate({ id: proposal.id, next: 'declined' })}
                  >
                    <X className="mr-2 h-4 w-4" aria-hidden="true" />
                    Decline
                  </Button>
                </div>
              )}

              {(proposal.status === 'accepted' || proposal.status === 'completed') && (
                <OutcomeCard
                  proposal={proposal}
                  existingOutcome={outcomeByProposalId.get(proposal.id) ?? null}
                />
              )}
            </Card>
          ))
        )}
      </main>

      <BottomNav />
    </div>
  );
}
