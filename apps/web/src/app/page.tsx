'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Bell,
  Brain,
  CalendarHeart,
  Compass,
  Footprints,
  HeartHandshake,
  Loader2,
  MoonStar,
  PawPrint,
  Sparkles,
  Target,
} from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';
import { BottomNav } from '@/components/bottom-nav';
import { FullScreenPostView } from '@/components/feed/full-screen-post-view';
import { PostCard } from '@/components/feed/post-card';
import { PWAInstallPrompt } from '@/components/pwa-install-prompt';
import { Button } from '@/components/ui/button';
import {
  type InsightRecommendation,
  insightsApi,
} from '@/lib/api/insights';
import { webSocialApi } from '@/lib/api/social';

const fallbackActions = [
  {
    href: '/discover',
    label: 'Find a match',
    description: 'Compatibility-first discovery',
    icon: Compass,
  },
  {
    href: '/events',
    label: 'Plan a meetup',
    description: 'Choose a comfortable shared setting',
    icon: CalendarHeart,
  },
  {
    href: '/activity',
    label: 'Log activity',
    description: 'Teach Woof about your shared routine',
    icon: Footprints,
  },
  {
    href: '/profile',
    label: 'Update preferences',
    description: 'Tell Woof what you are noticing',
    icon: Brain,
  },
];

const recommendationIcons = {
  activity: Footprints,
  enrichment: Sparkles,
  social: HeartHandshake,
  recovery: MoonStar,
  reflection: Brain,
  goal: Target,
} satisfies Record<InsightRecommendation['category'], typeof Footprints>;

export default function HomePage() {
  const queryClient = useQueryClient();
  const [fullScreenIndex, setFullScreenIndex] = useState<number | null>(null);

  const {
    data: insights,
    isLoading: insightsLoading,
    error: insightsError,
  } = useQuery({
    queryKey: ['insights', 'me'],
    queryFn: () => insightsApi.getMine(),
    retry: false,
  });

  const {
    data: posts = [],
    isLoading: feedLoading,
    error: feedError,
  } = useQuery({
    queryKey: ['feed'],
    queryFn: webSocialApi.getFeed,
  });

  const likeMutation = useMutation({
    mutationFn: ({ postId, isLiked }: { postId: string; isLiked: boolean }) =>
      isLiked ? webSocialApi.unlikePost(postId) : webSocialApi.likePost(postId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['feed'] });
    },
  });

  const handleLike = (postId: string) => {
    const post = posts.find((candidate) => candidate.id === postId);
    if (!post) return;
    likeMutation.mutate({ postId, isLiked: post.isLiked });
  };

  const handleRecommendationAccepted = (recommendation: InsightRecommendation) => {
    if (!insights) return;
    void insightsApi.feedback(insights.pet.id, recommendation, 'accepted');
  };

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <Link
            href="/"
            className="flex min-h-0 min-w-0 items-center gap-3"
            aria-label="Woof home"
          >
            <span className="brand-mark flex h-9 w-9 items-center justify-center rounded-xl">
              <PawPrint
                className="h-5 w-5 text-primary-foreground"
                aria-hidden="true"
              />
            </span>
            <span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Learn together
              </span>
              <span className="block text-lg font-bold tracking-tight">Woof</span>
            </span>
          </Link>

          <Button variant="ghost" size="icon" asChild className="relative rounded-xl">
            <Link href="/notifications" aria-label="Open notifications">
              <Bell className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 pb-6 pt-5">
        <section aria-labelledby="today-heading" className="animate-in">
          <div className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.09] via-card/80 to-secondary/[0.07] p-5 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="eyebrow">Today&apos;s learning loop</p>
                <h1
                  id="today-heading"
                  className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl"
                >
                  {insights
                    ? `What might help ${insights.pet.name} today?`
                    : 'What does your pet need today?'}
                </h1>
                <p className="mt-2 max-w-md text-sm leading-relaxed text-muted-foreground">
                  Woof combines the routines you record, the preferences you share, and
                  outcomes from real experiences to suggest useful next steps without
                  pretending certainty.
                </p>
              </div>
              {insights && (
                <div className="shrink-0 rounded-2xl border border-border/60 bg-background/60 px-3 py-2 text-right">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    Context confidence
                  </p>
                  <p className="mt-0.5 text-lg font-bold text-primary">
                    {Math.round(insights.algorithm.confidence * 100)}%
                  </p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-4">
            {insightsLoading ? (
              <div
                className="surface-soft flex min-h-40 items-center justify-center rounded-2xl"
                role="status"
              >
                <div className="text-center">
                  <Loader2
                    className="mx-auto h-6 w-6 animate-spin text-primary"
                    aria-hidden="true"
                  />
                  <p className="mt-2 text-sm text-muted-foreground">
                    Reading the recent routine…
                  </p>
                </div>
              </div>
            ) : insights && insights.recommendations.length > 0 ? (
              <div className="space-y-3">
                {insights.recommendations.slice(0, 3).map((recommendation, index) => {
                  const Icon = recommendationIcons[recommendation.category];
                  return (
                    <Link
                      key={recommendation.id}
                      href={recommendation.href}
                      onClick={() => handleRecommendationAccepted(recommendation)}
                      className="group surface-soft flex gap-4 rounded-2xl p-4 transition-colors hover:border-primary/30 hover:bg-primary/[0.045]"
                    >
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                              {index === 0 ? 'Best next step' : recommendation.category}
                            </p>
                            <h2 className="mt-1 font-semibold tracking-tight">
                              {recommendation.title}
                            </h2>
                          </div>
                          <span className="shrink-0 text-xs font-semibold text-primary">
                            {Math.round(recommendation.confidence * 100)}% context
                          </span>
                        </div>
                        <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                          {recommendation.reason}
                        </p>
                        <span className="mt-3 inline-flex text-sm font-semibold text-primary group-hover:text-primary/80">
                          {recommendation.actionLabel} →
                        </span>
                      </div>
                    </Link>
                  );
                })}
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3">
                {fallbackActions.map((action) => {
                  const Icon = action.icon;
                  return (
                    <Link
                      key={action.href}
                      href={action.href}
                      className="group surface-soft flex min-h-[122px] flex-col justify-between rounded-2xl p-4 transition-colors hover:border-primary/30 hover:bg-primary/[0.045]"
                    >
                      <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary transition-transform group-hover:-translate-y-0.5">
                        <Icon className="h-5 w-5" aria-hidden="true" />
                      </span>
                      <span className="mt-5">
                        <span className="block text-sm font-semibold">{action.label}</span>
                        <span className="mt-1 block text-xs leading-relaxed text-muted-foreground">
                          {action.description}
                        </span>
                      </span>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>

          {insights && (
            <div className="mt-6">
              <div className="mb-3 flex items-end justify-between gap-4">
                <div>
                  <p className="eyebrow">What Woof is learning</p>
                  <h2 className="mt-1 text-xl font-bold tracking-tight">
                    Your relationship, in signals
                  </h2>
                </div>
                <span className="text-[10px] text-muted-foreground">
                  Not a medical or bond score
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {insights.relationshipSignals.map((signal) => (
                  <div key={signal.key} className="surface-soft rounded-2xl p-4">
                    <div className="flex items-baseline justify-between gap-2">
                      <p className="text-sm font-semibold">{signal.label}</p>
                      <span className="text-sm font-bold text-primary">{signal.value}</span>
                    </div>
                    <div
                      className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted"
                      aria-hidden="true"
                    >
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${signal.value}%` }}
                      />
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      {signal.explanation}
                    </p>
                  </div>
                ))}
              </div>

              {insights.learningSummary.length > 0 && (
                <div className="mt-3 rounded-2xl border border-secondary/20 bg-secondary/[0.05] p-4">
                  <p className="text-xs font-semibold uppercase tracking-[0.14em] text-secondary-foreground">
                    Recent observations
                  </p>
                  <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-muted-foreground">
                    {insights.learningSummary.map((observation) => (
                      <li key={observation} className="flex gap-2">
                        <span className="text-primary" aria-hidden="true">
                          •
                        </span>
                        <span>{observation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {insightsError && !insights && (
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              Personalized guidance is unavailable right now, so Woof is showing the
              core actions instead. Your feed and account still work normally.
            </p>
          )}
        </section>

        <section aria-labelledby="feed-heading" className="mt-8">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="eyebrow">Community, not the finish line</p>
              <h2 id="feed-heading" className="mt-1 text-xl font-bold tracking-tight">
                Learn from your pack
              </h2>
            </div>
            <Link
              href="/discover"
              className="flex min-h-0 min-w-0 items-center gap-1 text-sm font-semibold text-primary hover:text-primary/80"
            >
              Discover
              <span aria-hidden="true">→</span>
            </Link>
          </div>

          <div className="overflow-hidden rounded-2xl border border-border/60 bg-card/55">
            {feedLoading ? (
              <div
                className="flex min-h-52 flex-col items-center justify-center gap-3 px-5 py-12"
                role="status"
              >
                <Loader2
                  className="h-7 w-7 animate-spin text-primary"
                  aria-hidden="true"
                />
                <p className="text-sm text-muted-foreground">
                  Gathering the latest from your pack…
                </p>
              </div>
            ) : feedError ? (
              <div className="flex min-h-52 flex-col items-center justify-center px-6 py-12 text-center">
                <div className="mb-4 flex h-11 w-11 items-center justify-center rounded-2xl bg-destructive/10 text-destructive">
                  <PawPrint className="h-5 w-5" aria-hidden="true" />
                </div>
                <h3 className="font-semibold">The feed could not load</h3>
                <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                  Your relationship-learning tools still work. Try the community feed
                  again when the connection settles.
                </p>
                <Button
                  variant="outline"
                  className="mt-5 bg-transparent"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['feed'] })}
                >
                  Try again
                </Button>
              </div>
            ) : posts.length === 0 ? (
              <div className="flex min-h-56 flex-col items-center justify-center px-6 py-12 text-center">
                <div className="brand-mark mb-4 flex h-12 w-12 items-center justify-center rounded-2xl">
                  <PawPrint
                    className="h-6 w-6 text-primary-foreground"
                    aria-hidden="true"
                  />
                </div>
                <h3 className="text-base font-semibold">Your pack is quiet for now</h3>
                <p className="mt-1 max-w-xs text-sm leading-relaxed text-muted-foreground">
                  Share a meaningful walk, play session, lesson, or park moment when
                  there is something worth remembering.
                </p>
                <div className="mt-5 flex flex-wrap justify-center gap-2">
                  <Button asChild>
                    <Link href="/discover">Find matches</Link>
                  </Button>
                  <Button variant="outline" asChild className="bg-transparent">
                    <Link href="/camera">Create a post</Link>
                  </Button>
                </div>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {posts.map((post, index) => (
                  <PostCard
                    key={post.id}
                    post={post}
                    onLike={handleLike}
                    onMediaClick={
                      post.mediaUrl ? () => setFullScreenIndex(index) : undefined
                    }
                  />
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      {fullScreenIndex !== null && (
        <FullScreenPostView
          posts={posts.filter((post) => Boolean(post.mediaUrl))}
          initialIndex={Math.max(
            0,
            posts
              .slice(0, fullScreenIndex)
              .filter((post) => Boolean(post.mediaUrl)).length,
          )}
          onClose={() => setFullScreenIndex(null)}
          onLike={handleLike}
        />
      )}

      <BottomNav />
      <PWAInstallPrompt />
    </div>
  );
}
