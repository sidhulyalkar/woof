'use client';

import { useQuery } from '@tanstack/react-query';
import { CloudOff, HeartHandshake, Loader2, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { conciergeApi } from '@/lib/api/concierge';

export function ConciergeBriefing({ petId, revision }: { petId: string; revision: string }) {
  const briefing = useQuery({
    queryKey: ['concierge', 'today', petId, revision],
    queryFn: () => conciergeApi.getToday(petId),
    retry: false,
    staleTime: 30_000,
  });

  if (briefing.isLoading) {
    return (
      <Card className="surface-soft mb-4 flex items-center gap-3 rounded-3xl p-4" role="status">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        </span>
        <div>
          <p className="text-sm font-semibold">Building today&apos;s context</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Reading existing dogOS evidence. No automatic actions are taken.
          </p>
        </div>
      </Card>
    );
  }

  if (briefing.isError || !briefing.data) return null;

  const data = briefing.data;
  const visibleSuggestions = data.suggestions.slice(0, 3);

  return (
    <section className="mb-5" aria-labelledby="concierge-briefing-heading">
      <Card className="rounded-3xl border-primary/20 bg-gradient-to-br from-primary/[0.07] via-card/95 to-secondary/[0.04] p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <p className="eyebrow">Concierge · today</p>
              <span className="rounded-full border border-border/70 bg-background/55 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                {data.context.pace.mode === 'GENTLE' ? 'Gentle pace' : 'Normal pace'}
              </span>
            </div>
            <h2 id="concierge-briefing-heading" className="mt-1 text-xl font-bold tracking-tight">
              {data.briefing.title}
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
              {data.briefing.summary}
            </p>
          </div>
        </div>

        {data.briefing.topQuest && (
          <div className="mt-4 rounded-2xl border border-border/70 bg-background/55 p-4">
            <div className="flex items-start gap-3">
              <HeartHandshake className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Why the deck leads here
                </p>
                <p className="mt-1 font-semibold">{data.briefing.topQuest.title}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {data.briefing.topQuest.reason}
                </p>
              </div>
            </div>
          </div>
        )}

        {visibleSuggestions.length > 0 && (
          <div className="mt-4 space-y-2">
            {visibleSuggestions.map((suggestion) => (
              <article key={suggestion.id} className="rounded-2xl border border-border/70 bg-card/65 p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold">{suggestion.title}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                      {suggestion.body}
                    </p>
                    <details className="mt-2 text-xs text-muted-foreground">
                      <summary className="cursor-pointer font-semibold text-foreground/80">
                        Why this is here
                      </summary>
                      <p className="mt-1 leading-relaxed">{suggestion.reason}</p>
                      <ul className="mt-1 space-y-1">
                        {suggestion.evidence.map((evidence, index) => (
                          <li key={`${suggestion.id}:${index}`}>• {evidence.label}</li>
                        ))}
                      </ul>
                    </details>
                  </div>
                  {suggestion.action && (
                    <Button variant="outline" size="sm" asChild className="shrink-0 bg-transparent">
                      <Link href={suggestion.action.href}>{suggestion.action.label}</Link>
                    </Button>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}

        <div className="mt-4 flex items-start gap-2 border-t border-border/60 pt-3 text-[11px] leading-relaxed text-muted-foreground">
          <CloudOff className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          <span>
            Live weather is not connected in this release. Concierge uses existing dogOS evidence
            only and never changes records or acts on your behalf.
          </span>
        </div>
      </Card>
    </section>
  );
}
