'use client';

import { useQuery } from '@tanstack/react-query';
import { ChevronRight, Loader2, PlugZap, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { Card } from '@/components/ui/card';
import { connectorsApi } from '@/lib/api/connectors';

export function ConnectedServicesSettings() {
  const dashboard = useQuery({
    queryKey: ['connectors'],
    queryFn: connectorsApi.getDashboard,
    retry: false,
    staleTime: 15_000,
  });

  const connected =
    dashboard.data?.providers.filter((provider) => provider.availability === 'CONNECTED').length ??
    0;
  const needsAttention =
    dashboard.data?.providers.filter((provider) => provider.availability === 'REAUTH_REQUIRED')
      .length ?? 0;

  return (
    <section className="space-y-3" aria-labelledby="connected-services-heading">
      <div>
        <p className="eyebrow">External context</p>
        <h2 id="connected-services-heading" className="mt-1 text-lg font-bold">
          Connected services
        </h2>
        <p className="mt-1 text-sm leading-6 text-muted-foreground">
          See which outside services can contribute sourced context without becoming a second dogOS
          source of truth.
        </p>
      </div>

      <Link
        href="/connectors"
        className="block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        <Card className="surface-soft flex items-center gap-3 rounded-2xl p-4 transition hover:border-primary/30">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <PlugZap className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="font-semibold">Manage connected services</p>
            {dashboard.isLoading ? (
              <p className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                Checking provider state
              </p>
            ) : dashboard.isError ? (
              <p className="mt-1 flex items-center gap-1.5 text-sm text-muted-foreground">
                <ShieldCheck className="h-3.5 w-3.5" aria-hidden="true" />
                Connector state unavailable
              </p>
            ) : needsAttention > 0 ? (
              <p className="mt-1 text-sm text-muted-foreground">
                {needsAttention} service{needsAttention === 1 ? '' : 's'} need reauthorization
              </p>
            ) : connected > 0 ? (
              <p className="mt-1 text-sm text-muted-foreground">
                {connected} verified service{connected === 1 ? '' : 's'} connected
              </p>
            ) : (
              <p className="mt-1 text-sm text-muted-foreground">
                Available integrations are currently partner-gated
              </p>
            )}
          </div>
          <ChevronRight className="h-5 w-5 shrink-0 text-muted-foreground" aria-hidden="true" />
        </Card>
      </Link>
    </section>
  );
}
