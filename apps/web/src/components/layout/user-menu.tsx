'use client';

import { useState } from 'react';
import { Trophy, User, LogOut } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { authApi } from '@/lib/api';
import { getInitials } from '@/lib/utils';

export function UserMenu() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);

  // TODO: Get user data from auth context/store
  const user = {
    name: 'Demo User',
    handle: '@demo',
    points: 2450,
    avatarUrl: null,
  };

  const handleLogout = () => {
    authApi.logout();
    router.push('/auth/login');
  };

  return (
    <div className="flex items-center space-x-4">
      {/* Points Display */}
      <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-primary/30 border border-accent/20">
        <Trophy className="w-4 h-4 text-yellow-400" />
        <span className="text-sm font-medium text-gray-200">{user.points.toLocaleString()} pts</span>
      </div>

      {/* User Avatar */}
      <div className="relative">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-10 h-10 rounded-full bg-gradient-to-br from-accent to-accent-600 flex items-center justify-center text-white font-medium text-sm hover:scale-105 transition-transform"
        >
          {user.avatarUrl ? (
            <img src={user.avatarUrl} alt={user.name} className="w-full h-full rounded-full object-cover" />
          ) : (
            <span>{getInitials(user.name)}</span>
          )}
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
            <div className="absolute right-0 mt-2 w-56 rounded-lg bg-surface border border-primary/20 shadow-xl z-50">
              <div className="p-3 border-b border-primary/20">
                <p className="font-medium text-gray-200">{user.name}</p>
                <p className="text-sm text-gray-400">{user.handle}</p>
              </div>
              <div className="p-1">
                <button
                  onClick={() => {
                    setIsOpen(false);
                    router.push('/profile');
                  }}
                  className="w-full flex items-center space-x-2 px-3 py-2 rounded-md text-gray-300 hover:bg-primary/30 hover:text-accent transition-colors"
                >
                  <User className="w-4 h-4" />
                  <span>Profile</span>
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full flex items-center space-x-2 px-3 py-2 rounded-md text-gray-300 hover:bg-primary/30 hover:text-red-400 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
