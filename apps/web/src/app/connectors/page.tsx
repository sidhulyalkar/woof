'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronLeft,
  Loader2,
  LockKeyhole,
  MapPinned,
  PlugZap,
  ShieldCheck,
  ShoppingBag,
  Stethoscope,
  Watch,
} from 'lucide-react';
import Link from 'next/link';
import { toast } from 'sonner';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  connectorsApi,
  type ConnectorDomain,
  type ConnectorProvider,
  type ConnectorProviderState,
} from '@/lib/api/connectors';

const capabilityLabels: Record<string, string> = {
  DAILY_ACTIVITY: 'Daily activity',
  DEVICE_STATUS: 'Device status',
  APPOINTMENT_IMPORT: 'Appointments',
  VACCINATION_IMPORT: 'Vaccinations',
  MEDICATION_REFERENCE: 'Medication instructions',
  DOCUMENT_REFERENCE: 'Source documents',
  CATALOG_REFERENCE: 'Product catalog',
  USER_APPROVED_HANDOFF: 'User-approved handoff',
};

const statusLabels = {
  PARTNER_REQUIRED: 'Partner access required',
  CONNECTED: 'Connected',
  REAUTH_REQUIRED: 'Reconnect required',
  REVOKED: 'Disconnected',
} as const;

function domainIcon(domain: ConnectorDomain) {
  if (domain === 'WEARABLE') return Watch;
  if (domain === 'VET') return Stethoscope;
  return ShoppingBag;
}

function providerNote(provider: ConnectorProvider) {
  if (provider === 'FI' || provider === 'TRACTIVE') {
    return 'Summary-only wearable context. Precise tracker location remains off.';
  }
  if (provider === 'VET_PARTNER') {
    return 'Imported records keep provider provenance. dogOS never computes a prescription or dose.';
  }
  return 'Catalog context can support a suggestion, but purchases always require your action.';
}

function providerStatusClass(provider: ConnectorProviderState) {
  if (provider.availability === 'CONNECTED') return 'bg-primary/10 text-primary';
  if (provider.availability === 'REAUTH_REQUIRED') return 'bg-amber-500/10 text-amber-700';
  return 'border border-border text-muted-foreground';
}

export default function ConnectorsPage() {
  const queryClient = useQueryClient();
  const dashboard = useQuery({
    queryKey: ['connectors'],
    queryFn: connectorsApi.getDashboard,
    retry: false,
  });

  const disconnect = useMutation({
    mutationFn: connectorsApi.disconnect,
    onSuccess: async (result) => {
      await queryClient.invalidateQueries({ queryKey: ['connectors'] });
      toast.success(`${result.provider} disconnected from dogOS`);
    },
    onError: () => toast.error('That service could not be disconnected. Please try again.'),
  });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <Button variant="ghost" size="icon" asChild className="rounded-xl">
            <Link href="/settings" aria-label="Back to settings">
              <ChevronLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
            <PlugZap className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
          <div>
            <p className="eyebrow">dogOS Connectors</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Connected services</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl space-y-6 px-4 py-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.1] via-card/90 to-secondary/[0.07] p-5 shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <ShieldCheck className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">Bring context, not control</p>
              <h2 className="mt-1 text-2xl font-bold tracking-tight">
                External services stay outside your dog&apos;s source of truth.
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Connectors can import bounded, sourced context through verified provider transports.
                They cannot silently rewrite your dog, activities, private memories, or Story.
              </p>
            </div>
          </div>
        </section>

        {dashboard.isLoading ? (
          <div className="flex min-h-[35vh] items-center justify-center" role="status">
            <Loader2 className="h-7 w-7 animate-spin text-primary" aria-hidden="true" />
            <span className="sr-only">Loading connected services</span>
          </div>
        ) : dashboard.isError || !dashboard.data ? (
          <Card className="surface-soft rounded-3xl p-6 text-center">
            <PlugZap className="mx-auto h-8 w-8 text-primary" aria-hidden="true" />
            <h2 className="mt-3 text-lg font-bold">Connected services are unavailable</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              The feature may be paused in this environment. Existing dogOS records are unchanged.
            </p>
            <Button className="mt-4" variant="outline" onClick={() => dashboard.refetch()}>
              Try again
            </Button>
          </Card>
        ) : (
          <>
            <section className="space-y-3" aria-labelledby="services-heading">
              <div>
                <p className="eyebrow">Provider access</p>
                <h2 id="services-heading" className="mt-1 text-xl font-bold tracking-tight">
                  Services
                </h2>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  A service only says Connected when dogOS has both verified connection metadata and
                  a usable authenticated credential. Partner-gated services stay visibly gated.
                </p>
              </div>

              <div className="space-y-3">
                {dashboard.data.providers.map((provider) => {
                  const Icon = domainIcon(provider.domain);
                  const canDisconnect =
                    provider.availability === 'CONNECTED' ||
                    provider.availability === 'REAUTH_REQUIRED';
                  return (
                    <Card key={provider.provider} className="surface-soft rounded-3xl p-4">
                      <div className="flex items-start gap-3">
                        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                          <Icon className="h-5 w-5" aria-hidden="true" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div>
                              <h3 className="font-semibold">{provider.label}</h3>
                              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                                {provider.domain === 'WEARABLE'
                                  ? 'Wearable'
                                  : provider.domain === 'VET'
                                    ? 'Veterinary'
                                    : 'Retail'}
                              </p>
                            </div>
                            <span
                              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide ${providerStatusClass(provider)}`}
                            >
                              {statusLabels[provider.availability]}
                            </span>
                          </div>

                          <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                            {providerNote(provider.provider)}
                          </p>

                          <div className="mt-3 flex flex-wrap gap-1.5">
                            {provider.capabilities.map((capability) => (
                              <span
                                key={capability}
                                className="rounded-full border border-border/70 bg-background/50 px-2 py-1 text-[10px] font-medium text-muted-foreground"
                              >
                                {capabilityLabels[capability] ?? capability}
                              </span>
                            ))}
                          </div>

                          {provider.availability === 'PARTNER_REQUIRED' && (
                            <p className="mt-3 rounded-2xl border border-border/70 bg-background/55 p-3 text-xs leading-relaxed text-muted-foreground">
                              No supported provider authorization flow is configured yet, so dogOS
                              will not offer a pretend Connect button.
                            </p>
                          )}

                          {provider.availability === 'REAUTH_REQUIRED' && (
                            <p className="mt-3 rounded-2xl border border-border/70 bg-background/55 p-3 text-xs leading-relaxed text-muted-foreground">
                              The stored provider credential is expired or could not be authenticated.
                              No new provider data will import until a verified reauthorization flow
                              becomes available.
                            </p>
                          )}

                          {canDisconnect && (
                            <Button
                              className="mt-3"
                              size="sm"
                              variant="outline"
                              disabled={disconnect.isPending}
                              onClick={() => disconnect.mutate(provider.provider)}
                            >
                              {disconnect.isPending && (
                                <Loader2
                                  className="mr-1.5 h-4 w-4 animate-spin"
                                  aria-hidden="true"
                                />
                              )}
                              Disconnect
                            </Button>
                          )}
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            </section>

            <section className="space-y-3" aria-labelledby="connector-boundaries-heading">
              <div>
                <p className="eyebrow">Trust boundary</p>
                <h2 id="connector-boundaries-heading" className="mt-1 text-xl font-bold">
                  What Connectors cannot do
                </h2>
              </div>
              <Card className="surface-soft grid grid-cols-1 gap-3 rounded-3xl p-4 sm:grid-cols-2">
                <Boundary
                  icon={MapPinned}
                  title="Import precise tracker GPS"
                  value="No"
                  detail="Location needs a separate explicit scope, retention policy, and deletion contract."
                />
                <Boundary
                  icon={ShieldCheck}
                  title="Rewrite canonical dog records"
                  value="No"
                  detail="Provider transport delegates to domain importers instead."
                />
                <Boundary
                  icon={ShoppingBag}
                  title="Place retail orders"
                  value="No"
                  detail="Retail connectors stop at user-approved handoff."
                />
                <Boundary
                  icon={LockKeyhole}
                  title="Store raw provider payloads"
                  value="No"
                  detail="Operational provenance stores normalized hashes and references only."
                />
              </Card>
            </section>

            <Card className="rounded-3xl border-primary/15 bg-primary/[0.04] p-4">
              <div className="flex items-start gap-3">
                <LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div>
                  <p className="font-semibold">Credential vault</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    {dashboard.data.credentialEncryptionConfigured
                      ? 'Connector credentials use authenticated encryption before persistence.'
                      : 'Credential encryption is not configured in this environment, so credential operations fail closed.'}
                  </p>
                </div>
              </div>
            </Card>
          </>
        )}
      </main>

      <BottomNav />
    </div>
  );
}

function Boundary({
  icon: Icon,
  title,
  value,
  detail,
}: {
  icon: typeof ShieldCheck;
  title: string;
  value: 'No';
  detail: string;
}) {
  return (
    <div className="rounded-2xl border border-border/70 bg-background/55 p-3">
      <div className="flex items-center gap-2">
        <Icon className="h-4 w-4 text-primary" aria-hidden="true" />
        <p className="text-xs font-semibold">{title}</p>
        <span className="ml-auto text-xs font-bold text-primary">{value}</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{detail}</p>
    </div>
  );
}
