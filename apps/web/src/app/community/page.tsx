'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Gamepad2,
  Globe2,
  HeartHandshake,
  Loader2,
  MapPinned,
  PawPrint,
  ShieldCheck,
  Sparkles,
  Trophy,
  Users,
} from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { ShareableMoments } from '@/components/social-adventure/shareable-moments';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { socialAdventureApi, type SocialAdventureReaction } from '@/lib/api/social-adventure';

const reactionCopy: Record<SocialAdventureReaction, string> = {
  NICE_READ: 'Nice read',
  GOOD_CALL: 'Good call',
  TRYING_THIS: 'Trying this',
  ADVENTURE_INSPIRATION: 'Adventure inspiration',
  CHEER: 'Cheer',
};

export default function CommunityPage() {
  const queryClient = useQueryClient();
  const me = useQuery({
    queryKey: ['social-adventure', 'me'],
    queryFn: socialAdventureApi.getMine,
    retry: false,
  });
  const leaderboard = useQuery({
    queryKey: ['social-adventure', 'leaderboard', 'global'],
    queryFn: socialAdventureApi.globalLeaderboard,
    retry: false,
  });
  const feed = useQuery({
    queryKey: ['social-adventure', 'feed'],
    queryFn: socialAdventureApi.feed,
    retry: false,
  });

  const preferenceMutation = useMutation({
    mutationFn: socialAdventureApi.updatePreferences,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'me'] }),
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'leaderboard'] }),
      ]);
    },
  });

  const reactionMutation = useMutation({
    mutationFn: ({
      shareId,
      reaction,
      remove,
    }: {
      shareId: string;
      reaction: SocialAdventureReaction;
      remove: boolean;
    }) =>
      remove
        ? socialAdventureApi.removeReaction(shareId, reaction)
        : socialAdventureApi.addReaction(shareId, reaction),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['social-adventure', 'feed'] }),
  });

  const loading = me.isLoading || leaderboard.isLoading || feed.isLoading;

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Users className="h-5 w-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Social Adventure
            </p>
            <h1 className="text-lg font-bold tracking-tight">Community</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/20 bg-gradient-to-br from-primary/[0.11] via-card/95 to-secondary/[0.06] p-5">
          <div className="flex items-start gap-3">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary text-primary-foreground">
              <Trophy className="h-5 w-5" aria-hidden="true" />
            </div>
            <div>
              <p className="eyebrow">You compete. Your dog doesn&apos;t.</p>
              <h2 className="mt-1 text-2xl font-bold tracking-tight">
                Get better at the human side.
              </h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Social Adventure rewards Human Skill games and varied, suitable Adventures.
                Distance, repetitions, posting, likes, health status, and missed days are worth zero
                league points.
              </p>
            </div>
          </div>

          {me.data && (
            <div className="mt-5 rounded-2xl border border-border/70 bg-background/55 p-4">
              <div className="flex items-end justify-between gap-3">
                <div>
                  <p className="text-xs font-semibold text-muted-foreground">This week</p>
                  <p className="mt-1 text-3xl font-black tracking-tight text-primary">
                    {me.data.score}
                    <span className="ml-1 text-sm font-semibold text-muted-foreground">
                      / {me.data.maxScore}
                    </span>
                  </p>
                </div>
                <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-bold text-primary">
                  Human + pair league
                </span>
              </div>
              <Progress className="mt-3 h-2" value={(me.data.score / me.data.maxScore) * 100} />
              <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                <div className="rounded-xl bg-card/70 p-3">
                  <p className="font-semibold">Human Skill</p>
                  <p className="mt-1 text-muted-foreground">
                    {me.data.components.humanSkill.score}/{me.data.components.humanSkill.maxScore}
                  </p>
                </div>
                <div className="rounded-xl bg-card/70 p-3">
                  <p className="font-semibold">Adventure variety</p>
                  <p className="mt-1 text-muted-foreground">
                    {me.data.components.adventureVariety.pathways.length} pathways explored
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="mt-4 grid grid-cols-2 gap-2">
            <Button asChild>
              <Link href="/arcade">
                <Gamepad2 className="mr-2 h-4 w-4" aria-hidden="true" />
                Skill Arcade
              </Link>
            </Button>
            <Button variant="outline" asChild className="bg-transparent">
              <Link href="/community/packs">
                <MapPinned className="mr-2 h-4 w-4" aria-hidden="true" />
                Local Packs
              </Link>
            </Button>
            <Button variant="outline" asChild className="col-span-2 bg-transparent">
              <Link href="/pack">
                <HeartHandshake className="mr-2 h-4 w-4" aria-hidden="true" />
                Cooperative Pack quests
              </Link>
            </Button>
          </div>
        </section>

        {loading && (
          <div className="flex min-h-40 items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
          </div>
        )}

        {!loading && leaderboard.data && me.data && (
          <section className="mt-7" aria-labelledby="global-league-heading">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="eyebrow">Opt-in league</p>
                <h2 id="global-league-heading" className="mt-1 text-xl font-bold tracking-tight">
                  Global human-skill league
                </h2>
              </div>
              <Globe2 className="mt-1 h-5 w-5 text-primary" aria-hidden="true" />
            </div>

            <div className="surface-soft mt-3 rounded-2xl p-4">
              <div className="flex items-start gap-3">
                <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
                <div className="min-w-0 flex-1">
                  <p className="font-semibold">
                    {me.data.preferences.globalLeaderboardOptIn
                      ? 'You are visible in the global league.'
                      : 'Your score is private by default.'}
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    Opting in publishes your handle and Social Adventure score only. It does not
                    publish pet health, route data, Daily Signals, or private notes.
                  </p>
                  <Button
                    variant={me.data.preferences.globalLeaderboardOptIn ? 'outline' : 'default'}
                    size="sm"
                    className="mt-3"
                    disabled={preferenceMutation.isPending}
                    onClick={() =>
                      preferenceMutation.mutate(!me.data.preferences.globalLeaderboardOptIn)
                    }
                  >
                    {me.data.preferences.globalLeaderboardOptIn
                      ? 'Make my rank private'
                      : 'Join global league'}
                  </Button>
                </div>
              </div>
            </div>

            <div className="mt-3 space-y-2">
              {leaderboard.data.entries.length === 0 ? (
                <div className="surface-soft rounded-2xl p-5 text-center">
                  <Trophy className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
                  <p className="mt-2 font-semibold">
                    The league is still gathering its first players.
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Private-by-default means an empty podium is allowed.
                  </p>
                </div>
              ) : (
                leaderboard.data.entries.slice(0, 10).map((entry) => (
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
                        {entry.components.humanSkill.score} skill ·{' '}
                        {entry.components.adventureVariety.pathways.length} pathways
                      </p>
                    </div>
                    <span className="text-lg font-black text-primary">{entry.score}</span>
                  </article>
                ))
              )}
            </div>
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              {leaderboard.data.disclaimer}
            </p>
          </section>
        )}

        <ShareableMoments />

        <section className="mt-8" aria-labelledby="community-feed-heading">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="eyebrow">Optional sharing</p>
              <h2 id="community-feed-heading" className="mt-1 text-xl font-bold tracking-tight">
                Adventure feed
              </h2>
            </div>
            <Sparkles className="h-5 w-5 text-primary" aria-hidden="true" />
          </div>

          {!feed.data || feed.data.posts.length === 0 ? (
            <div className="surface-soft mt-3 rounded-3xl p-6 text-center">
              <PawPrint className="mx-auto h-7 w-7 text-primary" aria-hidden="true" />
              <h3 className="mt-3 font-bold">No shared moments yet</h3>
              <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                Recent private Adventure moments appear above when they are eligible to share.
                Nothing is posted automatically.
              </p>
            </div>
          ) : (
            <div className="mt-3 space-y-4">
              {feed.data.posts.map((post) => (
                <article
                  key={post.shareId}
                  className="rounded-3xl border border-border/70 bg-card/70 p-5"
                >
                  <div className="flex items-start gap-3">
                    <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                      <PawPrint className="h-5 w-5" aria-hidden="true" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <p className="font-semibold">@{post.handle}</p>
                        {post.petName && (
                          <span className="text-xs text-muted-foreground">with {post.petName}</span>
                        )}
                      </div>
                      <p className="mt-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-primary">
                        {post.kind.replaceAll('_', ' ')}
                      </p>
                    </div>
                  </div>

                  <h3 className="mt-4 text-lg font-bold tracking-tight">{post.headline}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                    {post.summary}
                  </p>
                  {post.caption && post.caption !== post.summary && (
                    <p className="mt-3 rounded-2xl bg-background/55 p-3 text-sm leading-relaxed">
                      {post.caption}
                    </p>
                  )}

                  <div className="mt-4 flex flex-wrap gap-2">
                    {post.reactions.map((reaction) => (
                      <button
                        key={reaction.reaction}
                        type="button"
                        disabled={reactionMutation.isPending}
                        onClick={() =>
                          reactionMutation.mutate({
                            shareId: post.shareId,
                            reaction: reaction.reaction,
                            remove: reaction.mine,
                          })
                        }
                        className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition-colors ${
                          reaction.mine
                            ? 'border-primary/40 bg-primary/10 text-primary'
                            : 'border-border bg-background/40 text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        {reactionCopy[reaction.reaction]}
                        {reaction.count > 0 ? ` · ${reaction.count}` : ''}
                      </button>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>

      <BottomNav />
    </div>
  );
}
