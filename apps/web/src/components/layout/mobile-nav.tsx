'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Footprints, Trophy, Users, Home } from 'lucide-react';
import { cn } from '@/lib/utils';

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Pets', href: '/pets', icon: Footprints },
  { name: 'Activities', href: '/activities', icon: Activity },
  { name: 'Social', href: '/social', icon: Users },
  { name: 'Ranks', href: '/leaderboard', icon: Trophy },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-surface/95 backdrop-blur-sm border-t border-primary/20 z-50">
      <div className="grid grid-cols-5 h-16">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex flex-col items-center justify-center transition-all',
                isActive ? 'text-accent' : 'text-gray-400 hover:text-accent'
              )}
            >
              <Icon className={cn('w-5 h-5', isActive && 'scale-110')} />
              <span className="text-xs mt-1">{item.name}</span>
              {isActive && (
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-12 h-0.5 bg-accent rounded-t-full" />
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
