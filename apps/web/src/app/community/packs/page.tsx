'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Loader2, MapPinned, ShieldCheck, Trophy, Users } from 'lucide-react';
import Link from 'next/link';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { socialAdventureApi } from '@/lib/api/social-adventure';

export default function LocalPacksPage() {
  const queryClient = useQueryClient();
  const [selectedPackId, setSelectedPackId] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [regionKey, setRegionKey] = useState('');
  const [repairRegionKey, setRepairRegionKey] = useState('');

  const regions = useQuery({
    queryKey: ['social-adventure', 'regions'],
    queryFn: socialAdventureApi.regions,
    retry: false,
  });

  const packs = useQuery({
    queryKey: ['social-adventure', 'packs'],
    queryFn: socialAdventureApi.packs,
    retry: false,
  });

  useEffect(() => {
    const firstRegion = regions.data?.regions[0]?.id;
    if (!firstRegion) return;
    setRegionKey((current) => current || firstRegion);
    setRepairRegionKey((current) => current || firstRegion);
  }, [regions.data]);

  useEffect(() => {
    if (selectedPackId || !packs.data) return;
    setSelectedPackId(packs.data.packs.find((pack) => pack.joined)?.id ?? null);
  }, [packs.data, selectedPackId]);

  const selectedPack = useMemo(
    () => packs.data?.packs.find((pack) => pack.id === selectedPackId) ?? null,
    [packs.data, selectedPackId]
  );

  const leaderboard = useQuery({
    queryKey: ['social-adventure', 'packs', selectedPackId, 'leaderboard'],
    queryFn: () => socialAdventureApi.packLeaderboard(selectedPackId as string),
    enabled: Boolean(selectedPackId && selectedPack?.localityStatus === 'APPROVED'),
    retry: false,
  });

  const createMutation = useMutation({
    mutationFn: socialAdventureApi.createPack,
    onSuccess: async (created) => {
      setName('');
      setSelectedPackId(created.id);
      await queryClient.invalidateQueries({ queryKey: ['social-adventure', 'packs'] });
    },
  });

  const repairMutation = useMutation({
    mutationFn: ({ packId, approvedRegionId }: { packId: string; approvedRegionId: string }) =>
      socialAdventureApi.repairPackLocality(packId, approvedRegionId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['social-adventure', 'packs'] });
      await queryClient.invalidateQueries({
        queryKey: ['social-adventure', 'packs', selectedPackId, 'leaderboard'],
      });
    },
  });

  const joinMutation = useMutation({
    mutationFn: socialAdventureApi.joinPack,
    onSuccess: async (_, packId) => {
      setSelectedPackId(packId);
      await queryClient.invalidateQueries({ queryKey: ['social-adventure', 'packs'] });
    },
  });

  const submitPack = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (name.trim().length < 2 || !regionKey) return;
    createMutation.mutate({ name: name.trim(), regionKey });
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <MapPinned className="h-5 w-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Social Adventure
            </p>
            <h1 className="text-lg font-bold tracking-tight">Local Packs</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.1] via-card/95 to-secondary/[0.06] p-5">
          <p className="eyebrow">Local without tracking you</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight">
            Choose a broad community, never a coordinate.
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Pack locality now comes only from Woof&apos;s server-approved broad-area catalog. The app
            does not turn typed addresses, venues, home location, route endpoints, or live GPS into
            locality authority. Local ranks stay hidden until the Pack has enough active members.
          </p>
          <Button variant="outline" asChild className="mt-4 bg-transparent">
            <Link href="/community">← Back to Community</Link>
          </Button>
        </section>

        <section className="mt-6" aria-labelledby="pack-list-heading">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="eyebrow">Opt-in communities</p>
              <h2 id="pack-list-heading" className="mt-1 text-xl font-bold tracking-tight">
                Find a Pack
              </h2>
            </div>
            <Users className="h-5 w-5 text-primary" aria-hidden="true" />
          </div>

          {packs.isLoading ? (
            <div className="flex min-h-32 items-center justify-center" role="status">
              <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
            </div>
          ) : !packs.data?.packs.length ? (
            <div className="surface-soft mt-3 rounded-2xl p-5 text-center">
              <p className="font-semibold">No local Packs yet.</p>
              <p className="mt-1 text-sm text-muted-foreground">
                You can start the first approved broad-area Pack below.
              </p>
            </div>
          ) : (
            <div className="mt-3 space-y-2">
              {packs.data.packs.map((pack) => (
                <article
                  key={pack.id}
                  className={`surface-soft rounded-2xl p-4 ${selectedPackId === pack.id ? 'ring-2 ring-primary/25' : ''}`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <button
                      type="button"
                      onClick={() => setSelectedPackId(pack.id)}
                      className="min-w-0 flex-1 text-left"
                    >
                      <p className="font-bold">{pack.name}</p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {pack.coarseRegion?.displayName ?? 'Locality needs owner repair'} ·{' '}
                        {pack.memberCount} {pack.memberCount === 1 ? 'member' : 'members'}
                      </p>
                    </button>
                    {pack.joined ? (
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                        Joined
                      </span>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        className="bg-transparent"
                        disabled={joinMutation.isPending}
                        onClick={() => joinMutation.mutate(pack.id)}
                      >
                        Join
                      </Button>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        {selectedPack?.localityStatus === 'LEGACY_UNVERIFIED' && selectedPack.role === 'OWNER' && (
          <section className="mt-7 rounded-3xl border border-amber-300/50 bg-amber-50/50 p-5">
            <p className="eyebrow">Locality repair</p>
            <h2 className="mt-1 text-xl font-bold tracking-tight">Choose a broad area again</h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              Woof discarded the Pack&apos;s old free-form locality rather than assuming it was safe.
              Select one approved broad area to restore public discovery and local standings.
            </p>
            <select
              aria-label="Approved broad area for legacy Pack"
              value={repairRegionKey}
              onChange={(event) => setRepairRegionKey(event.target.value)}
              className="mt-4 w-full rounded-xl border border-border bg-background px-3 py-2.5"
            >
              {regions.data?.regions.map((region) => (
                <option key={region.id} value={region.id}>
                  {region.displayName}
                </option>
              ))}
            </select>
            <Button
              className="mt-3"
              disabled={!repairRegionKey || repairMutation.isPending}
              onClick={() =>
                repairMutation.mutate({
                  packId: selectedPack.id,
                  approvedRegionId: repairRegionKey,
                })
              }
            >
              {repairMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              )}
              Confirm broad area
            </Button>
          </section>
        )}

        {selectedPackId && selectedPack?.localityStatus === 'APPROVED' && (
          <section className="mt-7" aria-labelledby="local-league-heading">
            <div>
              <p className="eyebrow">Pack league</p>
              <h2 id="local-league-heading" className="mt-1 text-xl font-bold tracking-tight">
                Local human-skill standings
              </h2>
            </div>

            {leaderboard.isLoading ? (
              <div className="flex min-h-24 items-center justify-center" role="status">
                <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
              </div>
            ) : leaderboard.data && !leaderboard.data.cohortReady ? (
              <div className="surface-soft mt-3 rounded-2xl p-5">
                <div className="flex items-start gap-3">
                  <ShieldCheck
                    className="mt-0.5 h-5 w-5 shrink-0 text-primary"
                    aria-hidden="true"
                  />
                  <div>
                    <p className="font-semibold">Building a privacy-safe cohort</p>
                    <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                      {leaderboard.data.message}
                    </p>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {leaderboard.data.pack.memberCount}/{leaderboard.data.minimumCohort} active
                      members
                    </p>
                  </div>
                </div>
              </div>
            ) : leaderboard.data ? (
              <div className="mt-3 space-y-2">
                {leaderboard.data.entries.map((entry) => (
                  <article
                    key={entry.userId}
                    className="surface-soft flex items-center gap-3 rounded-2xl p-3"
                  >
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-sm font-black text-primary">
                      {entry.rank}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-semibold">@{entry.handle}</p>
                      <p className="text-xs text-muted-foreground">
                        {entry.components.humanSkill.score} human skill ·{' '}
                        {entry.components.adventureVariety.pathways.length} pathways
                      </p>
                    </div>
                    <span className="text-lg font-black text-primary">{entry.score}</span>
                  </article>
                ))}
              </div>
            ) : null}
          </section>
        )}

        <section className="mt-7 rounded-3xl border border-border/70 bg-card/60 p-5">
          <div className="flex items-start gap-3">
            <Trophy className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
            <div>
              <h2 className="font-bold">Start a broad-area Pack</h2>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Choose from the approved catalog. Woof does not accept an address, school, venue,
                coordinate, route, or exact meetup point as Pack locality.
              </p>
            </div>
          </div>

          <form className="mt-4 space-y-3" onSubmit={submitPack}>
            <label className="block text-sm font-semibold">
              Pack name
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                maxLength={64}
                placeholder="South Bay Adventure Pack"
                className="mt-1.5 w-full rounded-xl border border-border bg-background/60 px-3 py-2.5 font-normal outline-none focus:border-primary"
              />
            </label>
            <label className="block text-sm font-semibold">
              Broad area
              <select
                value={regionKey}
                onChange={(event) => setRegionKey(event.target.value)}
                disabled={regions.isLoading || !regions.data?.regions.length}
                className="mt-1.5 w-full rounded-xl border border-border bg-background/60 px-3 py-2.5 font-normal outline-none focus:border-primary"
              >
                {regions.data?.regions.map((region) => (
                  <option key={region.id} value={region.id}>
                    {region.displayName}
                  </option>
                ))}
              </select>
            </label>
            <Button type="submit" disabled={createMutation.isPending || !regionKey}>
              {createMutation.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              )}
              Create Pack
            </Button>
            {(createMutation.isError || regions.isError) && (
              <p className="text-sm text-destructive">
                {regions.isError
                  ? 'Approved broad areas are unavailable, so Woof will not guess a locality.'
                  : 'That Pack could not be created. Refresh the approved broad-area catalog and try again.'}
              </p>
            )}
          </form>
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
