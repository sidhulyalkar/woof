'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Clock3, Loader2, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { caregiverApi, type CaregiverGrant } from '@/lib/api/caregiver';

function expiryCopy(expiresAt: string) {
  const date = new Date(expiresAt);
  if (Number.isNaN(date.getTime())) return 'Expiry unavailable';
  return `Until ${date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  })}`;
}

function capabilityCopy(grant: CaregiverGrant) {
  return grant.capabilities.includes('LOG_OBSERVATION')
    ? 'View today + leave context-only observations'
    : 'View today only';
}

export function CaregiverAccessPanel() {
  const queryClient = useQueryClient();
  const received = useQuery({
    queryKey: ['caregiver', 'received'],
    queryFn: caregiverApi.received,
    retry: false,
  });
  const active = useQuery({
    queryKey: ['caregiver', 'active-pets'],
    queryFn: caregiverApi.activePets,
    retry: false,
  });

  const transition = useMutation({
    mutationFn: async ({ grantId, action }: { grantId: string; action: 'accept' | 'decline' }) =>
      action === 'accept' ? caregiverApi.accept(grantId) : caregiverApi.decline(grantId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['caregiver', 'received'] }),
        queryClient.invalidateQueries({ queryKey: ['caregiver', 'active-pets'] }),
      ]);
    },
  });

  const pending =
    received.data?.filter(
      (grant) => grant.effectiveStatus === 'PENDING_ACCEPTANCE' && !grant.relationshipBlocked
    ) ?? [];
  const activeGrants = active.data ?? [];

  if (received.isLoading || active.isLoading) {
    return (
      <section className="mt-6 rounded-3xl border border-border/70 bg-card/50 p-5" role="status">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          Checking temporary caregiver access…
        </div>
      </section>
    );
  }

  if (received.isError || active.isError) {
    return null;
  }

  if (pending.length === 0 && activeGrants.length === 0) {
    return null;
  }

  return (
    <section
      data-caregiver-access-panel
      className="mt-6 rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.08] via-card/95 to-secondary/[0.05] p-5"
    >
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
          <ShieldCheck className="h-5 w-5" aria-hidden="true" />
        </span>
        <div>
          <p className="eyebrow">Temporary care</p>
          <h2 className="mt-1 text-lg font-bold">Caregiver access</h2>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            A caregiver grant is pet-scoped and time-bounded. It never turns you into a household
            member or gives reward, medical, profile, connector, or history authority.
          </p>
        </div>
      </div>

      {pending.length > 0 && (
        <div className="mt-5 space-y-3" aria-label="Pending caregiver invitations">
          {pending.map((grant) => {
            const busy = transition.isPending && transition.variables?.grantId === grant.id;
            return (
              <article
                key={grant.id}
                data-caregiver-pending-grant={grant.id}
                className="rounded-2xl border border-border/70 bg-background/65 p-4"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-bold">Care for {grant.pet.name}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Invited by @{grant.issuerHandle ?? 'guardian'} · {capabilityCopy(grant)}
                    </p>
                  </div>
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground">
                    <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
                    {expiryCopy(grant.expiresAt)}
                  </span>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={transition.isPending}
                    onClick={() => transition.mutate({ grantId: grant.id, action: 'accept' })}
                  >
                    {busy && transition.variables?.action === 'accept' && (
                      <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden="true" />
                    )}
                    Accept temporary care
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="bg-transparent"
                    disabled={transition.isPending}
                    onClick={() => transition.mutate({ grantId: grant.id, action: 'decline' })}
                  >
                    Decline
                  </Button>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {activeGrants.length > 0 && (
        <div className="mt-5 space-y-3" aria-label="Active caregiver access">
          {activeGrants.map((grant) => (
            <article
              key={grant.id}
              data-caregiver-active-grant={grant.id}
              className="rounded-2xl border border-border/70 bg-background/65 p-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-bold">
                    {grant.pet.name} is available for temporary care
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Guardian @{grant.issuerHandle ?? 'guardian'} · {capabilityCopy(grant)}
                  </p>
                </div>
                <span className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground">
                  <Clock3 className="h-3.5 w-3.5" aria-hidden="true" />
                  {expiryCopy(grant.expiresAt)}
                </span>
              </div>
              <Button asChild size="sm" className="mt-4">
                <Link href={`/caregiver/pets/${grant.pet.id}`}>Open caregiver Today</Link>
              </Button>
            </article>
          ))}
        </div>
      )}

      {transition.isError && (
        <p className="mt-4 text-sm text-destructive" role="alert">
          That caregiver invitation is no longer available. Woof did not grant any pet authority.
        </p>
      )}
    </section>
  );
}
