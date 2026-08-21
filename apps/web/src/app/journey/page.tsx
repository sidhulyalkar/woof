'use client';

import { useQuery } from '@tanstack/react-query';
import { Camera, Image as ImageIcon, Loader2, Map, PawPrint, Sparkles, Star, Video } from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { adventureApi } from '@/lib/api/adventure';
import { mediaLibraryApi } from '@/lib/api/media-library';

export default function JourneyPage() {
  const adventure = useQuery({
    queryKey: ['adventure', 'me'],
    queryFn: () => adventureApi.getMine(),
    retry: false,
  });

  const library = useQuery({
    queryKey: ['media-library', 'journey', adventure.data?.pet.id],
    queryFn: () => mediaLibraryApi.library(adventure.data!.pet.id, { limit: 18 }),
    enabled: Boolean(adventure.data?.pet.id),
    retry: false,
  });

  const pet = adventure.data?.pet;
  const assets = library.data?.assets ?? [];
  const albums = library.data?.albums ?? [];

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
            <Map className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Real world → game world
            </p>
            <h1 className="text-lg font-bold tracking-tight">Journey</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.1] via-card/90 to-secondary/[0.07] p-5">
          <p className="eyebrow">Adventure Book</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight">
            {pet ? `${pet.name}'s world is becoming a story.` : 'Shared experiences become a story.'}
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Memories are the evidence trail and the scrapbook, not the objective. The real reward comes
            from the experience itself.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button asChild>
              <Link href="/activity">Start an adventure</Link>
            </Button>
            <Button variant="outline" asChild className="bg-transparent">
              <Link href="/library">
                <Camera className="mr-2 h-4 w-4" aria-hidden="true" />
                Open full library
              </Link>
            </Button>
          </div>
        </section>

        {adventure.isLoading || library.isLoading ? (
          <div className="flex min-h-56 items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
          </div>
        ) : (
          <>
            <section className="mt-6">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Collections</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">Stamps from your actual life</h2>
                </div>
                <span className="text-xs text-muted-foreground">{albums.length} collections</span>
              </div>

              {albums.length > 0 ? (
                <div className="mt-3 flex gap-3 overflow-x-auto pb-2">
                  {albums.slice(0, 10).map((album) => (
                    <Link
                      key={album.id}
                      href="/library"
                      className="surface-soft min-w-[148px] rounded-2xl p-4 transition-colors hover:border-primary/30"
                    >
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-lg" aria-hidden="true">
                        {album.icon || '🐾'}
                      </span>
                      <p className="mt-4 text-sm font-semibold">{album.name}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{album.count} memories</p>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="surface-soft mt-3 rounded-2xl p-5">
                  <p className="font-semibold">Your Adventure Book has room to grow.</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Keep a favorite walk, training win, first beach trip, or quiet recovery day when it is
                    worth remembering.
                  </p>
                </div>
              )}
            </section>

            <section className="mt-6">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="eyebrow">Recent pages</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">Memories, not proof chores</h2>
                </div>
                <Button variant="ghost" size="sm" asChild className="text-primary">
                  <Link href="/library">See all →</Link>
                </Button>
              </div>

              {assets.length > 0 ? (
                <div className="mt-3 grid grid-cols-3 gap-2">
                  {assets.slice(0, 9).map((asset) => (
                    <Link
                      key={asset.id}
                      href="/library"
                      className="group relative aspect-square overflow-hidden rounded-2xl border border-border/60 bg-muted"
                    >
                      {asset.url && asset.mediaType === 'image' ? (
                        <img
                          src={asset.url}
                          alt={asset.filename}
                          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
                        />
                      ) : (
                        <div className="flex h-full items-center justify-center text-muted-foreground">
                          {asset.mediaType === 'video' ? (
                            <Video className="h-6 w-6" aria-hidden="true" />
                          ) : (
                            <ImageIcon className="h-6 w-6" aria-hidden="true" />
                          )}
                        </div>
                      )}
                      {asset.favorite && (
                        <span className="absolute right-2 top-2 rounded-full bg-background/80 p-1 text-primary backdrop-blur-sm">
                          <Star className="h-3.5 w-3.5 fill-current" aria-hidden="true" />
                        </span>
                      )}
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="surface-soft mt-3 flex min-h-40 flex-col items-center justify-center rounded-2xl p-6 text-center">
                  <PawPrint className="h-7 w-7 text-primary" aria-hidden="true" />
                  <p className="mt-3 font-semibold">No kept memories yet</p>
                  <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                    Photos stay optional. A completed quest is valuable even when the phone never leaves your pocket.
                  </p>
                </div>
              )}
            </section>

            <section className="mt-6 rounded-3xl border border-secondary/20 bg-secondary/[0.05] p-5">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div>
                  <p className="font-semibold">What Journey will learn next</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Route semantics, favorite places, first-time locations, terrain, season, and paired
                    outcome history can turn future outings into real map stamps without making location
                    sharing mandatory.
                  </p>
                </div>
              </div>
            </section>
          </>
        )}
      </main>

      <BottomNav />
    </div>
  );
}
