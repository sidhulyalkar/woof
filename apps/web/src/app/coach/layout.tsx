'use client';

import { Clock3, FlaskConical, Images, Video } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

export default function CoachLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const observing = pathname.startsWith('/coach/observe');
  const history = pathname.startsWith('/coach/observe/history');
  const shadow = pathname.startsWith('/coach/observe/shadow');
  const primaryHref = observing ? (shadow ? '/coach/observe' : '/coach/observe/shadow') : '/coach/observe';
  const primaryLabel = observing ? (shadow ? 'Observe behavior' : 'Shadow Lab') : 'Observe behavior';
  const PrimaryIcon = observing && !shadow ? FlaskConical : Video;

  return (
    <>
      {children}
      <div className="fixed bottom-24 right-4 z-40 flex items-center gap-2">
        <Link
          href="/library"
          aria-label="Open private pet media library"
          className="flex h-11 w-11 items-center justify-center rounded-full border border-border/70 bg-background/95 text-muted-foreground shadow-lg backdrop-blur-xl transition-transform hover:-translate-y-0.5 hover:text-foreground focus-visible:-translate-y-0.5"
        >
          <Images className="h-4 w-4" aria-hidden="true" />
        </Link>
        {observing && !history && !shadow && (
          <Link
            href="/coach/observe/history"
            aria-label="Open behavior history"
            className="flex h-11 w-11 items-center justify-center rounded-full border border-border/70 bg-background/95 text-muted-foreground shadow-lg backdrop-blur-xl transition-transform hover:-translate-y-0.5 hover:text-foreground focus-visible:-translate-y-0.5"
          >
            <Clock3 className="h-4 w-4" aria-hidden="true" />
          </Link>
        )}
        <Link
          href={primaryHref}
          className="flex min-h-11 items-center gap-2 rounded-full border border-primary/20 bg-background/95 px-4 py-2 text-sm font-semibold text-primary shadow-lg backdrop-blur-xl transition-transform hover:-translate-y-0.5 focus-visible:-translate-y-0.5"
        >
          <PrimaryIcon className="h-4 w-4" aria-hidden="true" />
          {primaryLabel}
        </Link>
      </div>
    </>
  );
}
