'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, LocateFixed, MapPinOff, RefreshCw, ShieldCheck, SlidersHorizontal, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { DiscoverMapView } from '@/components/discover/discover-map-view';
import { FilterSheet } from '@/components/discover/filter-sheet';
import { MatchCard } from '@/components/discover/match-card';
import { PetSwitcher } from '@/components/pets/pet-switcher';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { getActivePetId, setActivePetId } from '@/lib/active-pet';
import { authApi, compatibilityApi } from '@/lib/api';
import { discoveryApi, type DiscoveryDistanceBand } from '@/lib/api/discovery';
import { useAuthStore } from '@/lib/stores/auth-store';

const DISTANCE_ORDER: Record<DiscoveryDistanceBand, number> = {
  WITHIN_2_5_KM: 0,
  WITHIN_5_KM: 1,
  WITHIN_10_KM: 2,
};

function requestBrowserLocation() {
  return new Promise<GeolocationPosition>((resolve, reject) => {
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      reject(new Error('Location is not available in this browser'));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, reject, {
      enableHighAccuracy: false,
      timeout: 10_000,
      maximumAge: 5 * 60 * 1000,
    });
  });
}

export default function DiscoverPage() {
  const cachedUser = useAuthStore((state) => state.user);
  const router = useRouter();
  const searchParams = useSearchParams();
  const queryClient = useQueryClient();
  const [filterOpen, setFilterOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('matches');
  const [selectedPetId, setSelectedPetId] = useState<string | null>(null);

  const profile = useQuery({
    queryKey: ['auth-profile'],
    queryFn: authApi.me,
    staleTime: 30_000,
  });

  const user = profile.data ?? cachedUser;
  const ownedPets = user?.pets ?? [];

  useEffect(() => {
    if (ownedPets.length === 0) {
      setSelectedPetId(null);
      return;
    }

    const requestedPetId = searchParams.get('pet');
    const rememberedPetId = getActivePetId();
    const requestedOwned = ownedPets.some((pet) => pet.id === requestedPetId);
    const rememberedOwned = ownedPets.some((pet) => pet.id === rememberedPetId);
    const nextPetId = requestedOwned
      ? requestedPetId
      : rememberedOwned
        ? rememberedPetId
        : ownedPets[0]?.id;

    if (!nextPetId) return;
    setActivePetId(nextPetId);
    setSelectedPetId((current) => (current === nextPetId ? current : nextPetId));
    if (requestedPetId !== nextPetId) {
      router.replace(`/discover?pet=${nextPetId}`, { scroll: false });
    }
  }, [ownedPets, router, searchParams]);

  const matches = useQuery({
    queryKey: ['recommendations', selectedPetId],
    queryFn: () => compatibilityApi.getRecommendations(selectedPetId!),
    enabled: Boolean(selectedPetId),
    staleTime: 60_000,
  });

  const location = useQuery({
    queryKey: ['discovery', 'location'],
    queryFn: discoveryApi.getLocationStatus,
    enabled: Boolean(user),
    staleTime: 30_000,
    retry: false,
  });

  const nearby = useQuery({
    queryKey: ['discovery', 'nearby', selectedPetId],
    queryFn: () => discoveryApi.getNearby(selectedPetId!),
    enabled: Boolean(selectedPetId) && location.data?.status === 'OPTED_IN',
    staleTime: 60_000,
    retry: false,
  });

  const enableLocation = useMutation({
    mutationFn: async () => {
      const position = await requestBrowserLocation();
      return discoveryApi.enableLocation(position.coords.latitude, position.coords.longitude);
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['discovery', 'location'] }),
        queryClient.invalidateQueries({ queryKey: ['discovery', 'nearby'] }),
      ]);
    },
  });

  const disableLocation = useMutation({
    mutationFn: discoveryApi.disableLocation,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['discovery', 'location'] }),
        queryClient.removeQueries({ queryKey: ['discovery', 'nearby'] }),
      ]);
    },
  });

  const distanceByPet = useMemo(
    () => new Map((nearby.data?.candidates ?? []).map((candidate) => [candidate.petId, candidate.distanceBand])),
    [nearby.data],
  );

  const orderedMatches = useMemo(() => {
    const currentMatches = matches.data ?? [];
    return [...currentMatches].sort((a, b) => {
      const aBand = distanceByPet.get(a.pet.id);
      const bBand = distanceByPet.get(b.pet.id);
      const aOrder = aBand ? DISTANCE_ORDER[aBand] : 99;
      const bOrder = bBand ? DISTANCE_ORDER[bBand] : 99;
      if (aOrder !== bOrder) return aOrder - bOrder;
      return b.compatibility.overall - a.compatibility.overall;
    });
  }, [distanceByPet, matches.data]);

  const isLoading = profile.isLoading || matches.isLoading;
  const locationStatus = location.data?.status ?? 'NOT_CONFIGURED';

  const choosePet = (petId: string) => {
    setSelectedPetId(petId);
    router.replace(`/discover?pet=${petId}`, { scroll: false });
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-xl items-center justify-between gap-4 px-4 py-4">
          <div>
            <p className="eyebrow">Compatibility, not popularity</p>
            <h1 className="mt-1 text-2xl font-bold tracking-tight">Discover</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              {activeTab === 'matches'
                ? selectedPetId
                  ? `${orderedMatches.length} explainable ${orderedMatches.length === 1 ? 'match' : 'matches'}`
                  : 'Add a dog to start matching'
                : 'Explore places and services without exposing dog locations'}
            </p>
          </div>
          <Button
            variant="outline"
            size="icon"
            aria-label="Open discovery filters"
            onClick={() => setFilterOpen(true)}
            className="shrink-0 bg-transparent"
          >
            <SlidersHorizontal className="h-5 w-5" aria-hidden="true" />
          </Button>
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <div className="sticky top-[89px] z-30 border-b border-border/50 bg-background/88 backdrop-blur-2xl">
          <TabsList className="mx-auto grid h-12 w-full max-w-xl grid-cols-2 bg-transparent px-4">
            <TabsTrigger
              value="matches"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
            >
              Compatible dogs
            </TabsTrigger>
            <TabsTrigger
              value="map"
              className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary"
            >
              Places & services
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="matches" className="mt-0">
          <main id="main-content" className="mx-auto max-w-xl space-y-4 px-4 py-5">
            {selectedPetId && (
              <Card className="surface-soft rounded-2xl p-4">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                    <Sparkles className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="text-sm font-semibold">One dog, one discovery context</h2>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      Compatibility uses the same active dog as Today. Learned scoring can rerank only when its release evidence is promoted; deterministic scoring remains the fallback.
                    </p>
                  </div>
                </div>
                <PetSwitcher currentPetId={selectedPetId} label="Finding friends for" onChange={choosePet} />
              </Card>
            )}

            {user && selectedPetId && (
              <Card className="rounded-2xl border-primary/15 bg-gradient-to-br from-primary/[0.06] via-card/95 to-card p-4">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                    {locationStatus === 'OPTED_IN' ? (
                      <ShieldCheck className="h-5 w-5" aria-hidden="true" />
                    ) : (
                      <LocateFixed className="h-5 w-5" aria-hidden="true" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <h2 className="text-sm font-semibold">
                      {locationStatus === 'OPTED_IN' ? 'Nearby context is on' : 'Make matches locally useful'}
                    </h2>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      {locationStatus === 'OPTED_IN'
                        ? 'Woof stores only a coarse roughly 2 km location cell. Other members never receive your coordinates or home location.'
                        : 'Opt in only when you want nearby context. Your precise browser coordinate is immediately reduced to a coarse cell and is not persisted.'}
                    </p>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  {locationStatus === 'OPTED_IN' ? (
                    <>
                      <Button
                        size="sm"
                        variant="outline"
                        className="bg-transparent"
                        disabled={enableLocation.isPending}
                        onClick={() => enableLocation.mutate()}
                      >
                        {enableLocation.isPending && (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                        )}
                        Refresh rough location
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        disabled={disableLocation.isPending}
                        onClick={() => disableLocation.mutate()}
                      >
                        Turn off nearby
                      </Button>
                    </>
                  ) : (
                    <Button size="sm" disabled={enableLocation.isPending} onClick={() => enableLocation.mutate()}>
                      {enableLocation.isPending ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                      ) : (
                        <LocateFixed className="mr-2 h-4 w-4" aria-hidden="true" />
                      )}
                      Use rough location
                    </Button>
                  )}
                </div>

                {enableLocation.isError && (
                  <p className="mt-3 flex items-start gap-2 text-xs text-destructive" role="alert">
                    <MapPinOff className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                    Location was not enabled. You can keep using profile-based compatibility without it.
                  </p>
                )}
                {locationStatus === 'OPTED_IN' && nearby.data && (
                  <p className="mt-3 text-xs text-muted-foreground">
                    {nearby.data.candidates.length} public candidate
                    {nearby.data.candidates.length === 1 ? '' : 's'} currently fall inside your coarse nearby window. Nearby matches are sorted first.
                  </p>
                )}
              </Card>
            )}

            {isLoading ? (
              <div className="flex min-h-72 flex-col items-center justify-center gap-3" role="status">
                <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
                <p className="text-sm text-muted-foreground">Loading your dog and ranking candidates…</p>
              </div>
            ) : profile.isError && !cachedUser ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">Your profile could not be refreshed</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Discovery needs the current dog list before it can rank relationships. Other authenticated surfaces remain available.
                </p>
                <Button variant="outline" className="mt-5 gap-2 bg-transparent" onClick={() => profile.refetch()}>
                  <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  Retry profile
                </Button>
              </div>
            ) : !selectedPetId ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">Start with your dog</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Woof needs a real dog profile before it can explain compatibility or local context.
                </p>
                <Button className="mt-5" asChild>
                  <Link href="/pets/new">Add your dog</Link>
                </Button>
              </div>
            ) : matches.isError ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">Recommendations are temporarily unavailable</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Discovery failed locally without blocking the rest of Woof. No substitute matches are invented.
                </p>
                <Button
                  variant="outline"
                  className="mt-5 gap-2 bg-transparent"
                  onClick={() => matches.refetch()}
                  disabled={matches.isFetching}
                >
                  {matches.isFetching ? (
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                  ) : (
                    <RefreshCw className="h-4 w-4" aria-hidden="true" />
                  )}
                  Retry
                </Button>
              </div>
            ) : orderedMatches.length === 0 ? (
              <div className="surface-soft flex min-h-72 flex-col items-center justify-center rounded-2xl px-6 text-center">
                <h2 className="text-lg font-semibold">No compatible candidates yet</h2>
                <p className="mt-2 max-w-sm text-sm leading-relaxed text-muted-foreground">
                  Woof will not fill an empty network with fake profiles. As real public members become eligible, explainable matches will appear here.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {orderedMatches.map((match) => (
                  <MatchCard
                    key={match.id}
                    match={match}
                    distanceBand={distanceByPet.get(match.pet.id)}
                  />
                ))}
                <div className="py-5 text-center">
                  <p className="text-sm text-muted-foreground">That is the current real candidate set.</p>
                  <Button
                    variant="ghost"
                    className="mt-2 gap-2"
                    onClick={() => matches.refetch()}
                    disabled={matches.isFetching}
                  >
                    {matches.isFetching ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    ) : (
                      <RefreshCw className="h-4 w-4" aria-hidden="true" />
                    )}
                    Refresh matches
                  </Button>
                </div>
              </div>
            )}
          </main>
        </TabsContent>

        <TabsContent value="map" className="mt-0">
          <DiscoverMapView />
        </TabsContent>
      </Tabs>

      <FilterSheet open={filterOpen} onOpenChange={setFilterOpen} />
      <BottomNav />
    </div>
  );
}
