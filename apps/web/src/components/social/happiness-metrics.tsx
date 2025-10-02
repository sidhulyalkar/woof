'use client';

import { Heart, Footprints, Sun, Users, Camera } from 'lucide-react';

const metrics = [
  { icon: Footprints, label: 'Walks', value: 12, target: 14, unit: 'this week', color: 'from-blue-500 to-blue-600' },
  { icon: Sun, label: 'Outdoor Time', value: 18, target: 20, unit: 'hours', color: 'from-yellow-500 to-yellow-600' },
  { icon: Users, label: 'Social Time', value: 6, target: 8, unit: 'sessions', color: 'from-green-500 to-green-600' },
  { icon: Camera, label: 'Photos Shared', value: 8, target: 10, unit: 'this week', color: 'from-purple-500 to-purple-600' },
];

export function HappinessMetrics() {
  return (
    <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden">
      <div className="p-6">
        <h3 className="flex items-center gap-2 font-heading font-semibold text-gray-100 mb-4">
          <Heart className="w-5 h-5 text-red-400 fill-red-400" />
          Your Happiness Metrics
        </h3>

        <div className="grid grid-cols-2 gap-4">
          {metrics.map((metric, index) => {
            const Icon = metric.icon;
            const progress = (metric.value / metric.target) * 100;

            return (
              <div key={index} className="bg-primary/30 rounded-lg p-4 border border-primary/20">
                <div className="flex items-center justify-between mb-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-br ${metric.color}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-sm text-gray-400">
                    {metric.value}/{metric.target}
                  </span>
                </div>
                <p className="font-medium text-gray-100 mb-2">{metric.label}</p>
                <div className="h-2 bg-surface/50 rounded-full overflow-hidden mb-1">
                  <div
                    className={`h-full bg-gradient-to-r ${metric.color} rounded-full transition-all duration-500`}
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-xs text-gray-400">{metric.unit}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
