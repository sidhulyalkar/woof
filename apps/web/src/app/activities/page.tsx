'use client';

import { useQuery } from '@tanstack/react-query';
import { Plus, Activity, Footprints, Clock, Flame } from 'lucide-react';
import { activitiesApi } from '@/lib/api';
import { formatDistance, formatDuration, formatRelativeTime } from '@/lib/utils';

export default function ActivitiesPage() {
  const { data: activities, isLoading } = useQuery({
    queryKey: ['activities'],
    queryFn: activitiesApi.getAllActivities,
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-heading font-bold text-gray-100">Activities</h1>
        <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-accent to-accent-600 text-white rounded-lg hover:shadow-lg hover:shadow-accent/20 transition-all">
          <Plus className="w-4 h-4" />
          <span>Log Activity</span>
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-24 bg-surface/50 rounded-lg animate-pulse" />
          ))}
        </div>
      ) : activities && activities.length > 0 ? (
        <div className="space-y-3">
          {activities.map((activity: any) => (
            <ActivityCard key={activity.id} activity={activity} />
          ))}
        </div>
      ) : (
        <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg p-12 text-center">
          <Activity className="w-12 h-12 text-gray-500 mx-auto mb-4" />
          <h3 className="font-heading font-semibold text-gray-300 mb-2">No activities yet</h3>
          <p className="text-gray-400 mb-4">Start tracking your adventures!</p>
          <button className="px-6 py-2 bg-gradient-to-r from-accent to-accent-600 text-white rounded-lg hover:shadow-lg hover:shadow-accent/20 transition-all">
            Log Activity
          </button>
        </div>
      )}
    </div>
  );
}

function ActivityCard({ activity }: { activity: any }) {
  const getActivityIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'walk':
        return Footprints;
      case 'run':
        return Activity;
      default:
        return Activity;
    }
  };

  const Icon = getActivityIcon(activity.type);

  return (
    <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg p-4 hover:border-accent/30 transition-all">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-accent to-accent-600 flex items-center justify-center">
            <Icon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="font-heading font-semibold text-gray-100 capitalize">{activity.type}</h3>
            <p className="text-sm text-gray-400">
              {activity.pet?.name} • {formatRelativeTime(new Date(activity.createdAt))}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-6">
          {activity.distance && (
            <div className="text-center">
              <div className="flex items-center space-x-1 text-gray-100">
                <Footprints className="w-4 h-4" />
                <span className="font-semibold">{formatDistance(activity.distance)}</span>
              </div>
              <p className="text-xs text-gray-400">Distance</p>
            </div>
          )}
          {activity.duration && (
            <div className="text-center">
              <div className="flex items-center space-x-1 text-gray-100">
                <Clock className="w-4 h-4" />
                <span className="font-semibold">{formatDuration(activity.duration)}</span>
              </div>
              <p className="text-xs text-gray-400">Duration</p>
            </div>
          )}
          {activity.caloriesBurned && (
            <div className="text-center">
              <div className="flex items-center space-x-1 text-gray-100">
                <Flame className="w-4 h-4" />
                <span className="font-semibold">{activity.caloriesBurned}</span>
              </div>
              <p className="text-xs text-gray-400">Calories</p>
            </div>
          )}
        </div>
      </div>

      {activity.notes && (
        <p className="mt-3 text-sm text-gray-300 pl-16">{activity.notes}</p>
      )}
    </div>
  );
}
