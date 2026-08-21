'use client';

import { Brain, Compass, HeartPulse, Home, User } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';

const navItems = [
  { href: '/', icon: Home, label: 'Home' },
  { href: '/discover', icon: Compass, label: 'Discover' },
  { href: '/coach', icon: Brain, label: 'Coach', isSpecial: true },
  { href: '/health', icon: HeartPulse, label: 'Health' },
  { href: '/profile', icon: User, label: 'Profile' },
];

export function BottomNav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Primary navigation"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-border/60 bg-background/92 pb-safe backdrop-blur-2xl"
    >
      <div className="mx-auto flex h-[68px] max-w-xl items-center justify-around px-3">
        {navItems.map((item) => {
          const isActive = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          const Icon = item.icon;

          if (item.isSpecial) {
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-label="Open Woof Coach"
                aria-current={isActive ? 'page' : undefined}
                className={cn(
                  'group -mt-5 flex h-14 w-14 items-center justify-center rounded-2xl border brand-mark text-primary-foreground transition-transform hover:-translate-y-0.5 focus-visible:-translate-y-0.5',
                  isActive ? 'border-primary/50 ring-4 ring-primary/10' : 'border-primary/20'
                )}
              >
                <Icon className="h-6 w-6 transition-transform group-hover:scale-105" aria-hidden="true" />
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
                'group relative flex h-full flex-1 flex-col items-center justify-center gap-1 rounded-xl px-1 transition-colors',
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
              <span className="text-[11px] font-semibold tracking-wide">{item.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
