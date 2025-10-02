'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Footprints, Trophy, Users } from 'lucide-react';
import { cn } from '@/lib/utils';
import { UserMenu } from './user-menu';

const navigation = [
  { name: 'Dashboard', href: '/', icon: Activity },
  { name: 'Pets', href: '/pets', icon: Footprints },
  { name: 'Activities', href: '/activities', icon: Activity },
  { name: 'Social', href: '/social', icon: Users },
  { name: 'Leaderboard', href: '/leaderboard', icon: Trophy },
];

export function Header() {
  const pathname = usePathname();

  return (
    <header className="bg-surface/95 backdrop-blur-sm border-b border-primary/20 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2 group">
            <div className="relative">
              <Footprints className="w-8 h-8 text-accent transition-all group-hover:scale-110" />
              <div className="absolute -inset-1 bg-accent/20 rounded-full blur-md opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <span className="text-xl font-heading font-bold bg-gradient-to-r from-accent to-accent-300 bg-clip-text text-transparent">
              Woof
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                    isActive
                      ? 'bg-accent/10 text-accent'
                      : 'text-gray-300 hover:text-accent hover:bg-accent/5'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* User Menu */}
          <UserMenu />
        </div>
      </div>
    </header>
  );
}
