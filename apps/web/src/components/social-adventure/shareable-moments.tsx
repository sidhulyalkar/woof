'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, Loader2, Share2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { socialAdventureApi } from '@/lib/api/social-adventure';

export function ShareableMoments() {
  const queryClient = useQueryClient();
  const candidates = useQuery({
    queryKey: ['social-adventure', 'share-candidates'],
    queryFn: socialAdventureApi.shareCandidates,
    retry: false,
  });
  const shareMutation = useMutation({
    mutationFn: (sourceId: string) =>
      socialAdventureApi.createShare({
        sourceType: 'CARE_EVENT',
        sourceId,
        visibility: 'PUBLIC',
      }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'share-candidates'] }),
        queryClient.invalidateQueries({ queryKey: ['social-adventure', 'feed'] }),
      ]);
    },
  });

  if (candidates.isLoading) {
    return (
      <div className="mt-3 flex min-h-24 items-center justify-center" role="status">
        <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
      </div>
    );
  }

  if (!candidates.data?.candidates.length) return null;

  return (
    <section className="mt-8" aria-labelledby="shareable-moments-heading">
      <div>
        <p className="eyebrow">Private until you choose</p>
        <h2 id="shareable-moments-heading" className="mt-1 text-xl font-bold tracking-tight">
          Recent moments you could share
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
          These previews come from your canonical Adventure history. Private notes and detailed outcome
          context are not copied into the feed card.
        </p>
      </div>

      <div className="mt-3 space-y-3">
        {candidates.data.candidates.map((candidate) => (
          <article key={candidate.sourceId} className="surface-soft rounded-3xl p-5">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-primary">
                  {candidate.kind.replaceAll('_', ' ')}
                </p>
                <h3 className="mt-1 font-bold">{candidate.headline}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {candidate.summary}
                </p>
              </div>
              {candidate.petName && (
                <span className="shrink-0 rounded-full bg-primary/10 px-2.5 py-1 text-xs font-bold text-primary">
                  {candidate.petName}
                </span>
              )}
            </div>
            <Button
              size="sm"
              className="mt-4"
              disabled={shareMutation.isPending}
              onClick={() => shareMutation.mutate(candidate.sourceId)}
            >
              {shareMutation.isPending && shareMutation.variables === candidate.sourceId ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              ) : shareMutation.isSuccess && shareMutation.variables === candidate.sourceId ? (
                <Check className="mr-2 h-4 w-4" aria-hidden="true" />
              ) : (
                <Share2 className="mr-2 h-4 w-4" aria-hidden="true" />
              )}
              Share with Community
            </Button>
          </article>
        ))}
      </div>
    </section>
  );
}
