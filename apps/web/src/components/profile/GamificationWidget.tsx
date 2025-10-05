'use client';

import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Trophy, Flame, Star, Award, Zap } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

interface GamificationSummary {
  points: number;
  badges: string[];
  badgeCount: number;
  streak: number;
  lastActivity: string | null;
}

export function GamificationWidget() {
  const { data, isLoading } = useQuery<GamificationSummary>({
    queryKey: ['gamification-summary'],
    queryFn: () => apiClient.get<GamificationSummary>('/gamification/me/summary'),
  });

  const getBadgeIcon = (badgeType: string) => {
    switch (badgeType) {
      case 'first_match':
        return '🤝';
      case 'first_meetup':
        return '🎉';
      case 'super_social':
        return '🌟';
      case 'event_host':
        return '📅';
      case 'verified_owner':
        return '✅';
      case 'early_adopter':
        return '🚀';
      case 'streak_master':
        return '🔥';
      case 'community_builder':
        return '🏘️';
      default:
        return '🏅';
    }
  };

  const getBadgeLabel = (badgeType: string) => {
    return badgeType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (isLoading) {
    return (
      <Card className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-muted rounded w-1/2"></div>
          <div className="h-20 bg-muted rounded"></div>
        </div>
      </Card>
    );
  }

  if (!data) return null;

  return (
    <Card className="p-6 bg-gradient-to-br from-accent/10 to-primary/10 border-accent/20">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold flex items-center gap-2">
          <Trophy className="w-5 h-5 text-accent" />
          Achievements
        </h3>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        {/* Points */}
        <div className="text-center p-4 rounded-lg bg-white/50 backdrop-blur">
          <div className="text-2xl font-bold text-accent">{data.points}</div>
          <div className="text-xs text-muted-foreground flex items-center justify-center gap-1 mt-1">
            <Star className="w-3 h-3" />
            Points
          </div>
        </div>

        {/* Badges */}
        <div className="text-center p-4 rounded-lg bg-white/50 backdrop-blur">
          <div className="text-2xl font-bold text-accent">{data.badgeCount}</div>
          <div className="text-xs text-muted-foreground flex items-center justify-center gap-1 mt-1">
            <Award className="w-3 h-3" />
            Badges
          </div>
        </div>

        {/* Streak */}
        <div className="text-center p-4 rounded-lg bg-white/50 backdrop-blur">
          <div className="text-2xl font-bold text-accent flex items-center justify-center gap-1">
            {data.streak}
            {data.streak > 0 && <Flame className="w-5 h-5 text-orange-500" />}
          </div>
          <div className="text-xs text-muted-foreground flex items-center justify-center gap-1 mt-1">
            <Zap className="w-3 h-3" />
            Week Streak
          </div>
        </div>
      </div>

      {/* Badges Display */}
      {data.badges && data.badges.length > 0 && (
        <div>
          <h4 className="text-sm font-semibold mb-3">Your Badges</h4>
          <div className="flex flex-wrap gap-2">
            {data.badges.map((badge, idx) => (
              <Badge
                key={idx}
                variant="secondary"
                className="text-sm px-3 py-1 bg-white/70 backdrop-blur"
              >
                <span className="mr-1">{getBadgeIcon(badge)}</span>
                {getBadgeLabel(badge)}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {/* Streak Encouragement */}
      {data.streak > 0 && (
        <div className="mt-4 p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
          <p className="text-sm text-center">
            <Flame className="inline w-4 h-4 text-orange-500 mr-1" />
            <span className="font-semibold">{data.streak} week{data.streak > 1 ? 's' : ''} active!</span>
            <span className="text-muted-foreground ml-1">Keep it up!</span>
          </p>
        </div>
      )}

      {/* No Achievements Yet */}
      {data.points === 0 && data.badgeCount === 0 && (
        <div className="text-center py-4">
          <Trophy className="w-12 h-12 mx-auto mb-2 text-muted-foreground opacity-50" />
          <p className="text-sm text-muted-foreground">
            Start meeting dogs to earn points and badges!
          </p>
        </div>
      )}
    </Card>
  );
}
