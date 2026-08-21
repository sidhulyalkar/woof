'use client';

import { Video } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

export default function CoachLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const observing = pathname.startsWith('/coach/observe');

  return (
    <>
      {children}
      {!observing && (
        <Link
          href="/coach/observe"
          className="fixed bottom-24 right-4 z-40 flex min-h-11 items-center gap-2 rounded-full border border-primary/20 bg-background/95 px-4 py-2 text-sm font-semibold text-primary shadow-lg backdrop-blur-xl transition-transform hover:-translate-y-0.5 focus-visible:-translate-y-0.5"
        >
          <Video className="h-4 w-4" aria-hidden="true" />
          Observe behavior
        </Link>
      )}
    </>
  );
}
