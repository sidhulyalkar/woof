'use client';

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Check, Images, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { mediaLibraryApi, type UploadMediaInput } from '@/lib/api/media-library';

export function KeepMediaButton({
  petId,
  media,
  source,
  tags,
  linkedObservationId,
  capturedAt,
}: {
  petId: string;
  media: File | Blob;
  source: UploadMediaInput['source'];
  tags: string[];
  linkedObservationId?: string | null;
  capturedAt?: string;
}) {
  const queryClient = useQueryClient();
  const [saved, setSaved] = useState(false);
  const mutation = useMutation({
    mutationFn: () =>
      mediaLibraryApi.uploadMedia({
        petId,
        media,
        source,
        tags,
        linkedObservationId: linkedObservationId || undefined,
        capturedAt,
      }),
    onSuccess: () => {
      setSaved(true);
      void queryClient.invalidateQueries({ queryKey: ['media-library', petId] });
    },
  });

  if (saved) {
    return (
      <div
        role="status"
        className="flex min-h-11 items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] px-3 text-sm font-medium text-emerald-700 dark:text-emerald-300"
      >
        <Check className="h-4 w-4" aria-hidden="true" />
        Original kept privately
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <Button
        type="button"
        variant="outline"
        className="w-full"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
      >
        {mutation.isPending ? (
          <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
        ) : (
          <Images className="mr-2 h-4 w-4" aria-hidden="true" />
        )}
        Keep original in private library
      </Button>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Analysis stays transient unless you choose this. The private copy can later appear in albums,
        exports, and longitudinal comparisons.
      </p>
      {mutation.isError && (
        <p role="alert" className="text-xs text-destructive">
          {mutation.error instanceof Error
            ? mutation.error.message
            : 'Woof could not save this original.'}
        </p>
      )}
    </div>
  );
}
