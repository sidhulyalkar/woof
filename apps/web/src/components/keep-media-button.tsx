'use client';

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, Loader2, LockKeyhole, Save } from 'lucide-react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { mediaLibraryApi, type UploadMediaInput } from '@/lib/api/media-library';

export type KeepMediaButtonProps = {
  petId: string;
  media: File | Blob;
  source: Extract<UploadMediaInput['source'], 'behavior-vision' | 'health-lens'>;
  linkedObservationId?: string | null;
  tags?: string[];
  filename?: string;
};

function inferredFilename(media: File | Blob, source: KeepMediaButtonProps['source']) {
  if (media instanceof File && media.name) return media.name;
  const extension = media.type.startsWith('video/') ? 'webm' : media.type === 'image/png' ? 'png' : 'jpg';
  return `${source}-${Date.now()}.${extension}`;
}

export function KeepMediaButton({
  petId,
  media,
  source,
  linkedObservationId,
  tags = [],
  filename,
}: KeepMediaButtonProps) {
  const queryClient = useQueryClient();
  const save = useMutation({
    mutationFn: () =>
      mediaLibraryApi.uploadMedia({
        petId,
        media,
        filename: filename ?? inferredFilename(media, source),
        source,
        linkedObservationId: linkedObservationId ?? undefined,
        tags,
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['media-library', petId] });
    },
  });

  if (save.isSuccess) {
    return (
      <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.06] p-4">
        <div className="flex items-start gap-3">
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-sm font-semibold">Original saved privately</p>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              This copy now lives in the private pet library. The analysis record and original remain
              separate so you can delete either intentionally.
            </p>
            <Link href="/library" className="mt-2 inline-flex min-h-8 items-center text-xs font-semibold text-primary">
              Open private library
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-border/60 bg-background/45 p-4">
      <div className="flex items-start gap-3">
        <LockKeyhole className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold">Keep the original?</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Analysis is transient by default. Save this exact photo or video only if you want it in
            your private longitudinal pet record.
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="mt-3 min-h-11"
            disabled={save.isPending}
            onClick={() => save.mutate()}
          >
            {save.isPending ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <Save className="mr-2 h-4 w-4" aria-hidden="true" />
            )}
            {save.isPending ? 'Saving privately…' : 'Keep original privately'}
          </Button>
          {save.isError && (
            <p role="alert" className="mt-2 text-xs text-destructive">
              {save.error instanceof Error
                ? save.error.message
                : 'Woof could not save this original. The analysis remains available.'}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
