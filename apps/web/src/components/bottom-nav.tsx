'use client';

import { useQuery } from '@tanstack/react-query';
import {
  Brain,
  ClipboardCheck,
  Compass,
  Gamepad2,
  Map,
  PawPrint,
  Sparkles,
  Users,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { companionApi } from '@/lib/api/companion';
import { cn } from '@/lib/utils';

const petNavItems = [
  { href: '/', icon: PawPrint, label: 'Today' },
  { href: '/compass', icon: Compass, label: 'Compass' },
  { href: '/journey', icon: Map, label: 'Story', isSpecial: true },
  { href: '/autopilot', icon: Sparkles, label: 'Auto' },
  { href: '/coach', icon: Brain, label: 'Coach' },
  { href: '/community', icon: Users, label: 'Community' },
];

const companionNavItems = [
  { href: '/', icon: PawPrint, label: 'Today' },
  { href: '/arcade', icon: Gamepad2, label: 'Arcade' },
  { href: '/community', icon: Users, label: 'Community' },
  { href: '/companion/readiness', icon: ClipboardCheck, label: 'Readiness' },
];

export function BottomNav() {
  const pathname = usePathname();
  const companion = useQuery({
    queryKey: ['companion', 'state'],
    queryFn: companionApi.state,
    retry: false,
    staleTime: 60_000,
    // Navigation observes account authority but does not own its resolution.
    // In particular, mounting the fail-closed nav after a state error must not
    // refetch the same failed query and bounce the root router back to loading.
    refetchOnMount: false,
  });

  // Fail closed for pet-only navigation. Until account state is positively
  // resolved as PET_TODAY, only surfaces that are valid without a pet appear.
  const navItems = companion.data?.landing === 'PET_TODAY' ? petNavItems : companionNavItems;

  return (
    <nav
      aria-label="Primary navigation"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-border/60 bg-background/92 pb-safe backdrop-blur-2xl"
    >
      <div className="mx-auto flex h-[68px] max-w-xl items-center justify-around px-2">
        {navItems.map((item) => {
          const isActive = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          const Icon = item.icon;

          if ('isSpecial' in item && item.isSpecial) {
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-label="Open Our Story"
                aria-current={isActive ? 'page' : undefined}
                className={cn(
                  'group -mt-5 flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border brand-mark text-primary-foreground transition-transform hover:-translate-y-0.5 focus-visible:-translate-y-0.5',
                  isActive ? 'border-primary/50 ring-4 ring-primary/10' : 'border-primary/20'
                )}
              >
                <Icon
                  className="h-6 w-6 transition-transform group-hover:scale-105"
                  aria-hidden="true"
                />
                <span className="sr-only">{item.label}</span>
              </Link>
            );
          }

          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? 'page' : undefined}
              className={cn(
                'group relative flex h-full min-w-0 flex-1 flex-col items-center justify-center gap-1 rounded-xl px-0.5 transition-colors',
                isActive ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <span
                className={cn(
                  'absolute top-1.5 h-1 w-1 rounded-full bg-primary transition-opacity',
                  isActive ? 'opacity-100' : 'opacity-0'
                )}
                aria-hidden="true"
              />
              <Icon className="h-5 w-5" aria-hidden="true" />
              <span className="text-[10px] font-semibold tracking-wide sm:text-[11px]">
                {item.label}
              </span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
