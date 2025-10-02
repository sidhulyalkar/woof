'use client';

import { TrendingUp } from 'lucide-react';

interface WeeklyProgressProps {
  activities: any[];
}

export function WeeklyProgress({ activities }: WeeklyProgressProps) {
  // Calculate stats from activities
  const totalDistance = activities.reduce((sum, act) => sum + (act.distance || 0), 0);
  const totalActivities = activities.length;
  const weeklyProgress = Math.min(Math.round((totalActivities / 20) * 100), 100);

  return (
    <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden">
      <div className="p-6">
        <h3 className="flex items-center gap-2 font-heading font-semibold text-gray-100 mb-4">
          <TrendingUp className="w-5 h-5 text-accent" />
          Weekly Progress
        </h3>

        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium text-gray-300">Monthly Goal</span>
            <span className="text-sm text-gray-400">{weeklyProgress}% Complete</span>
          </div>

          <div className="h-3 bg-primary/30 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent to-accent-600 rounded-full transition-all duration-500"
              style={{ width: `${weeklyProgress}%` }}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-accent/10 rounded-lg border border-accent/20">
              <p className="text-2xl font-heading font-bold text-accent">
                {(totalDistance / 1000).toFixed(1)}
              </p>
              <p className="text-sm text-gray-400">km walked</p>
            </div>
            <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-500/20">
              <p className="text-2xl font-heading font-bold text-green-400">{totalActivities}</p>
              <p className="text-sm text-gray-400">activities</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
