'use client';

import { useQuery } from '@tanstack/react-query';
import {
  Brain,
  Footprints,
  Heart,
  HeartHandshake,
  HeartPulse,
  Loader2,
  MoonStar,
  PawPrint,
  ShieldCheck,
  Sparkles,
  TreePine,
} from 'lucide-react';
import Link from 'next/link';
import { BottomNav } from '@/components/bottom-nav';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { adventureApi, type WellbeingPathway } from '@/lib/api/adventure';

const pathwayMeta: Record<
  WellbeingPathway,
  { icon: typeof PawPrint; description: string; href: string; action: string }
> = {
  MOVE: {
    icon: Footprints,
    description: 'Movement and conditioning that fit the individual dog and the current day.',
    href: '/activity',
    action: 'Activity',
  },
  EXPLORE: {
    icon: TreePine,
    description: 'Nature, novelty, sniffing, and sensory exploration without turning miles into the goal.',
    href: '/journey',
    action: 'Journey',
  },
  ENRICH: {
    icon: Sparkles,
    description: 'Searching, scent work, puzzles, foraging, and other species-appropriate enrichment.',
    href: '/',
    action: 'Quest deck',
  },
  LEARN: {
    icon: Brain,
    description: 'Reward-based communication, life skills, and cooperative-care practice.',
    href: '/coach',
    action: 'Coach',
  },
  CONNECT: {
    icon: HeartHandshake,
    description: 'Comfortable social experiences where space, choice, and good matches matter more than volume.',
    href: '/pack',
    action: 'Pack',
  },
  CARE: {
    icon: ShieldCheck,
    description: 'Private preventive-care support. Medical severity is never a game score.',
    href: '/health',
    action: 'Health & care',
  },
  RECOVER: {
    icon: MoonStar,
    description: 'Rest, decompression, and easier days that count as real progress.',
    href: '/',
    action: 'Recovery quests',
  },
  BOND: {
    icon: Heart,
    description: 'Shared rituals and experiences that both halves of the team actually enjoy.',
    href: '/journey',
    action: 'Memories',
  },
};

export default function CompassPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['adventure', 'me'],
    queryFn: () => adventureApi.getMine(),
    retry: false,
  });

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <PawPrint className="h-5 w-5" aria-hidden="true" />
          </span>
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              Eight pathways
            </p>
            <h1 className="text-lg font-bold tracking-tight">Pawprint Compass</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-xl px-4 pb-8 pt-5">
        <section className="rounded-3xl border border-primary/15 bg-gradient-to-br from-primary/[0.09] via-card/85 to-secondary/[0.06] p-5">
          <p className="eyebrow">A map, not a grade</p>
          <h2 className="mt-1 text-2xl font-bold tracking-tight">
            Notice what has had room lately.
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            The Compass shows recent opportunity coverage. A full ring does not mean “healthy,” and an
            empty one does not mean “bad owner.” Different dogs, seasons, restrictions, and life stages
            should produce different shapes.
          </p>
        </section>

        {isLoading ? (
          <div className="flex min-h-64 items-center justify-center" role="status">
            <Loader2 className="h-6 w-6 animate-spin text-primary" aria-hidden="true" />
          </div>
        ) : error || !data ? (
          <div className="surface-soft mt-4 rounded-2xl p-5 text-center">
            <p className="font-semibold">Compass data is unavailable.</p>
            <p className="mt-1 text-sm text-muted-foreground">The Adventure migration may still need to be applied.</p>
          </div>
        ) : (
          <>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {data.compass.map((item) => {
                const meta = pathwayMeta[item.pathway];
                const Icon = meta.icon;
                return (
                  <article key={item.pathway} className="surface-soft rounded-3xl p-5">
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3">
                        <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                          <Icon className="h-5 w-5" aria-hidden="true" />
                        </span>
                        <div>
                          <h3 className="font-bold">{item.label}</h3>
                          <p className="text-xs text-muted-foreground">{item.xp} pathway XP</p>
                        </div>
                      </div>
                      <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                        {item.recentDays}d
                      </span>
                    </div>
                    <Progress className="mt-4 h-2" value={item.coverage} />
                    <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{meta.description}</p>
                    <Button variant="ghost" size="sm" asChild className="mt-3 px-0 text-primary">
                      <Link href={meta.href}>{meta.action} →</Link>
                    </Button>
                  </article>
                );
              })}
            </div>

            <section className="mt-5 rounded-3xl border border-border/60 bg-card/55 p-5">
              <div className="flex items-center gap-3">
                <HeartPulse className="h-5 w-5 text-primary" aria-hidden="true" />
                <div>
                  <p className="font-semibold">Care has a hard boundary</p>
                  <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                    Preventive routines can appear on the Compass. Emergency or illness flows do not award
                    XP, show confetti, or become social competition.
                  </p>
                </div>
              </div>
            </section>

            <p className="mt-5 text-center text-xs leading-relaxed text-muted-foreground">{data.disclaimer}</p>
          </>
        )}
      </main>
      <BottomNav />
    </div>
  );
}
