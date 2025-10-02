'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Trophy, MessageCircle, User, Home } from 'lucide-react';

const navigation = [
  { name: 'feed', href: '/', icon: Home },
  { name: 'activity', href: '/activities', icon: Activity },
  { name: 'leaderboard', href: '/leaderboard', icon: Trophy },
  { name: 'messages', href: '/social', icon: MessageCircle },
  { name: 'profile', href: '/profile', icon: User },
];

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="glass-card border-t border-border/20 px-4 py-3 mx-4 mb-4 rounded-xl">
      <div className="flex justify-around items-center max-w-md mx-auto">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href ||
            (item.href !== '/' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              href={item.href}
              className={`relative p-3 rounded-xl transition-all duration-300 ${
                isActive
                  ? 'bg-accent text-white shadow-lg scale-110'
                  : 'text-muted-foreground hover:text-foreground hover:scale-105'
              }`}
            >
              <Icon size={24} />
              {item.name === 'messages' && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-destructive rounded-full ring-2 ring-background animate-pulse"></div>
              )}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
