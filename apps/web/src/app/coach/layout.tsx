'use client';

import { Clock3, Video } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ReactNode } from 'react';

export default function CoachLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const observing = pathname.startsWith('/coach/observe');
  const history = pathname.startsWith('/coach/observe/history');
  const href = observing && !history ? '/coach/observe/history' : '/coach/observe';
  const label = observing && !history ? 'Behavior history' : 'Observe behavior';
  const Icon = observing && !history ? Clock3 : Video;

  return (
    <>
      {children}
      <Link
        href={href}
        className="fixed bottom-24 right-4 z-40 flex min-h-11 items-center gap-2 rounded-full border border-primary/20 bg-background/95 px-4 py-2 text-sm font-semibold text-primary shadow-lg backdrop-blur-xl transition-transform hover:-translate-y-0.5 focus-visible:-translate-y-0.5"
      >
        <Icon className="h-4 w-4" aria-hidden="true" />
        {label}
      </Link>
    </>
  );
}