'use client';

import { useQuery } from '@tanstack/react-query';
import { Trophy, MapPin } from 'lucide-react';
import { socialApi } from '@/lib/api';
import { getInitials } from '@/lib/utils';

export default function LeaderboardPage() {
  const { data: leaderboard, isLoading } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: socialApi.getLeaderboard,
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-6">
        {/* Global Leaderboard */}
        <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden">
          <div className="p-6">
            <h3 className="flex items-center gap-2 font-heading font-semibold text-gray-100 mb-1">
              <Trophy className="w-5 h-5 text-yellow-400" />
              Global Leaderboard
            </h3>
            <p className="text-sm text-gray-400 mb-4">Top pet owners this week</p>

            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-16 bg-primary/20 rounded-lg animate-pulse" />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {leaderboard?.map((entry: any, index: number) => (
                  <LeaderboardEntry
                    key={entry.userId}
                    rank={index + 1}
                    entry={entry}
                    isCurrentUser={entry.isCurrentUser}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Local Leaderboard */}
        <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden">
          <div className="p-6">
            <h3 className="flex items-center gap-2 font-heading font-semibold text-gray-100 mb-1">
              <MapPin className="w-5 h-5 text-accent" />
              Local Leaderboard
            </h3>
            <p className="text-sm text-gray-400 mb-4">Top pet owners in your area</p>

            {isLoading ? (
              <div className="space-y-3">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-16 bg-primary/20 rounded-lg animate-pulse" />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {leaderboard?.slice(0, 5).map((entry: any, index: number) => (
                  <LeaderboardEntry
                    key={entry.userId}
                    rank={index + 1}
                    entry={entry}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface LeaderboardEntryProps {
  rank: number;
  entry: {
    userId: string;
    handle: string;
    avatarUrl?: string;
    petName: string;
    totalPoints: number;
  };
  isCurrentUser?: boolean;
}

function LeaderboardEntry({ rank, entry, isCurrentUser }: LeaderboardEntryProps) {
  const getRankColor = (rank: number) => {
    if (rank === 1) return 'bg-yellow-500 text-white';
    if (rank === 2) return 'bg-gray-400 text-white';
    if (rank === 3) return 'bg-orange-600 text-white';
    return 'bg-primary/40 text-gray-300 border border-primary/30';
  };

  return (
    <div
      className={`flex items-center justify-between p-3 rounded-lg transition-all ${
        isCurrentUser
          ? 'bg-accent/10 border border-accent/30'
          : 'hover:bg-primary/20 border border-transparent'
      }`}
    >
      <div className="flex items-center space-x-3">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${getRankColor(rank)}`}>
          {rank}
        </div>
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-accent to-accent-600 flex items-center justify-center text-white font-medium text-sm">
          {entry.avatarUrl ? (
            <img src={entry.avatarUrl} alt={entry.handle} className="w-full h-full rounded-full object-cover" />
          ) : (
            <span>{getInitials(entry.handle)}</span>
          )}
        </div>
        <div>
          <p className="font-medium text-gray-100">
            {entry.handle}
            {isCurrentUser && <span className="ml-2 text-xs text-accent">(You)</span>}
          </p>
          <p className="text-sm text-gray-400">{entry.petName}</p>
        </div>
      </div>
      <div className="text-right">
        <p className="font-bold text-gray-100">{entry.totalPoints.toLocaleString()}</p>
        <p className="text-sm text-gray-400">points</p>
      </div>
    </div>
  );
}
