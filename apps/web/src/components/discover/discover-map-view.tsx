'use client';

import { useQuery } from '@tanstack/react-query';
import { Loader2, MapPin, RefreshCw, ShieldCheck, Star } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { servicesApi } from '@/lib/api';

export function DiscoverMapView() {
  const services = useQuery({
    queryKey: ['services', 'discovery'],
    queryFn: () => servicesApi.getServices(),
    staleTime: 5 * 60_000,
    retry: false,
  });

  return (
    <div className="mx-auto max-w-xl space-y-4 px-4 py-5">
      <Card className="surface-soft rounded-2xl p-4">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
          <div>
            <h2 className="text-sm font-semibold">Places are not dog pins</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              This surface shows real service-provider records. Woof does not place members or dogs
              on a public map, and nearby matching never returns another household&apos;s
              coordinates.
            </p>
          </div>
        </div>
      </Card>

      {services.isLoading ? (
        <div className="flex min-h-64 items-center justify-center gap-3" role="status">
          <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
          <span className="text-sm text-muted-foreground">Loading available services…</span>
        </div>
      ) : services.isError ? (
        <Card className="surface-soft rounded-2xl p-6 text-center">
          <h2 className="font-semibold">Places and services are temporarily unavailable</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Woof will not substitute invented businesses or fake reviews while the service directory
            cannot be read.
          </p>
          <Button
            variant="outline"
            className="mt-4 gap-2 bg-transparent"
            onClick={() => services.refetch()}
          >
            <RefreshCw className="h-4 w-4" aria-hidden="true" />
            Try again
          </Button>
        </Card>
      ) : (services.data?.length ?? 0) === 0 ? (
        <Card className="surface-soft rounded-2xl p-6 text-center">
          <MapPin className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
          <h2 className="mt-3 font-semibold">No verified service records yet</h2>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
            This stays empty until real providers are available. Discovery matches continue to work
            independently.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {services.data?.map((service) => (
            <Card key={service.id} className="rounded-2xl p-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="font-semibold">{service.name}</h3>
                    {service.verified && (
                      <span className="rounded-full bg-primary/10 px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-primary">
                        Verified
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-xs capitalize text-muted-foreground">
                    {service.serviceType.replace('-', ' ')}
                  </p>
                  {service.bio && (
                    <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                      {service.bio}
                    </p>
                  )}
                  <p className="mt-3 flex items-start gap-2 text-xs text-muted-foreground">
                    <MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    {service.location.address}
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <span className="inline-flex items-center gap-1 text-sm font-semibold">
                    <Star className="h-4 w-4" aria-hidden="true" />
                    {service.rating.toFixed(1)}
                  </span>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {service.reviewCount} review{service.reviewCount === 1 ? '' : 's'}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
