'use client';

import { useState } from 'react';
import { Play, Pause, Square, MapPin, Timer, Footprints, Flame, Heart, TrendingUp } from 'lucide-react';
import { useSessionStore } from '@/store/session';
import { useActivities } from '@/lib/api/hooks';
import { formatDistanceToNow } from 'date-fns';

type ActivityType = 'WALK' | 'RUN' | 'PLAY' | 'HIKE';

export function ActivityScreen() {
  const { user, pets } = useSessionStore();
  const { data: activities, isLoading } = useActivities();
  const [isTracking, setIsTracking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [selectedActivityType, setSelectedActivityType] = useState<ActivityType>('WALK');
  const [selectedPet, setSelectedPet] = useState<string | null>(pets[0]?.id || null);

  const [currentStats, setCurrentStats] = useState({
    duration: 0,
    distance: 0,
    calories: 0,
    steps: 0,
  });

  const handleStartActivity = () => {
    setIsTracking(true);
    setIsPaused(false);
  };

  const handlePauseActivity = () => {
    setIsPaused(!isPaused);
  };

  const handleStopActivity = () => {
    setIsTracking(false);
    setIsPaused(false);
    setCurrentStats({ duration: 0, distance: 0, calories: 0, steps: 0 });
  };

  const activityTypes: { type: ActivityType; icon: any; label: string; color: string }[] = [
    { type: 'WALK', icon: Footprints, label: 'Walk', color: 'bg-blue-500' },
    { type: 'RUN', icon: TrendingUp, label: 'Run', color: 'bg-orange-500' },
    { type: 'PLAY', icon: Heart, label: 'Play', color: 'bg-pink-500' },
    { type: 'HIKE', icon: MapPin, label: 'Hike', color: 'bg-green-500' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 border-b border-gray-200/60 shadow-sm">
        <div className="max-w-2xl mx-auto flex items-center justify-between px-4 h-14">
          <h1 className="text-lg font-semibold text-gray-900">Activity</h1>
        </div>
      </header>

      <main className="pb-20 pt-3">
        <div className="max-w-2xl mx-auto px-4 space-y-3">
          {isTracking ? (
            <div className="bg-white/80 backdrop-blur-md border border-gray-200/60 rounded-2xl shadow-sm p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  {(() => {
                    const activity = activityTypes.find(a => a.type === selectedActivityType);
                    const Icon = activity?.icon;
                    return (
                      <>
                        <div className={`w-10 h-10 rounded-full ${activity?.color} flex items-center justify-center`}>
                          {Icon && <Icon className="h-5 w-5 text-white" />}
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">{selectedActivityType}</h3>
                          {selectedPet && pets.find(p => p.id === selectedPet) && (
                            <p className="text-xs text-gray-600">with {pets.find(p => p.id === selectedPet)?.name}</p>
                          )}
                        </div>
                      </>
                    );
                  })()}
                </div>
                <div className="flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${isPaused ? 'bg-yellow-500' : 'bg-green-500'} animate-pulse`} />
                  <span className="text-xs font-medium text-gray-600">{isPaused ? 'Paused' : 'Active'}</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-gray-50 rounded-xl p-4 text-center">
                  <Timer className="h-5 w-5 text-blue-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-gray-900">{Math.floor(currentStats.duration / 60)}:{(currentStats.duration % 60).toString().padStart(2, '0')}</p>
                  <p className="text-xs text-gray-600 mt-1">Duration</p>
                </div>
                <div className="bg-gray-50 rounded-xl p-4 text-center">
                  <MapPin className="h-5 w-5 text-green-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-gray-900">{currentStats.distance.toFixed(2)}</p>
                  <p className="text-xs text-gray-600 mt-1">Distance (km)</p>
                </div>
                <div className="bg-gray-50 rounded-xl p-4 text-center">
                  <Flame className="h-5 w-5 text-orange-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-gray-900">{currentStats.calories}</p>
                  <p className="text-xs text-gray-600 mt-1">Calories</p>
                </div>
                <div className="bg-gray-50 rounded-xl p-4 text-center">
                  <Footprints className="h-5 w-5 text-purple-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-gray-900">{currentStats.steps}</p>
                  <p className="text-xs text-gray-600 mt-1">Steps</p>
                </div>
              </div>

              <div className="flex gap-3">
                <button
                  onClick={handlePauseActivity}
                  className="flex-1 py-3 bg-gray-100 hover:bg-gray-200 rounded-xl font-medium text-gray-900 transition-colors flex items-center justify-center gap-2"
                >
                  {isPaused ? <Play className="h-5 w-5" /> : <Pause className="h-5 w-5" />}
                  {isPaused ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={handleStopActivity}
                  className="flex-1 py-3 bg-red-500 hover:bg-red-600 rounded-xl font-medium text-white transition-colors flex items-center justify-center gap-2"
                >
                  <Square className="h-5 w-5" />
                  Stop
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-white/80 backdrop-blur-md border border-gray-200/60 rounded-2xl shadow-sm p-6">
              <h3 className="text-base font-semibold text-gray-900 mb-4">Start Activity</h3>

              <div className="grid grid-cols-4 gap-2 mb-4">
                {activityTypes.map((activity) => {
                  const Icon = activity.icon;
                  const isSelected = selectedActivityType === activity.type;
                  return (
                    <button
                      key={activity.type}
                      onClick={() => setSelectedActivityType(activity.type)}
                      className={`p-3 rounded-xl border-2 transition-all ${
                        isSelected
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 bg-white hover:border-gray-300'
                      }`}
                    >
                      <div className={`w-8 h-8 rounded-full ${activity.color} flex items-center justify-center mx-auto mb-1`}>
                        <Icon className="h-4 w-4 text-white" />
                      </div>
                      <p className={`text-xs font-medium ${isSelected ? 'text-blue-700' : 'text-gray-700'}`}>
                        {activity.label}
                      </p>
                    </button>
                  );
                })}
              </div>

              {pets.length > 0 && (
                <div className="mb-4">
                  <label className="text-sm font-medium text-gray-700 mb-2 block">Activity with</label>
                  <select
                    value={selectedPet || ''}
                    onChange={(e) => setSelectedPet(e.target.value || null)}
                    className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Just me</option>
                    {pets.map((pet) => (
                      <option key={pet.id} value={pet.id}>
                        {pet.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <button
                onClick={handleStartActivity}
                className="w-full py-3 bg-blue-500 hover:bg-blue-600 rounded-xl font-semibold text-white shadow-md hover:shadow-lg transition-all flex items-center justify-center gap-2"
              >
                <Play className="h-5 w-5" />
                Start {selectedActivityType}
              </button>
            </div>
          )}

          <div className="bg-white/80 backdrop-blur-md border border-gray-200/60 rounded-2xl shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200/60">
              <h3 className="text-base font-semibold text-gray-900">Recent Activities</h3>
            </div>
            <div className="divide-y divide-gray-100">
              {isLoading ? (
                <div className="p-8 text-center text-sm text-gray-600">Loading...</div>
              ) : activities && activities.length > 0 ? (
                activities.slice(0, 5).map((activity: any) => {
                  const activityType = activityTypes.find(a => a.type === activity.type);
                  const Icon = activityType?.icon;
                  return (
                    <div key={activity.id} className="px-6 py-4 hover:bg-gray-50/50 transition-colors">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className={`w-10 h-10 rounded-full ${activityType?.color || 'bg-gray-400'} flex items-center justify-center`}>
                            {Icon && <Icon className="h-5 w-5 text-white" />}
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-900">{activity.type}</p>
                            <p className="text-xs text-gray-600">
                              {formatDistanceToNow(new Date(activity.startedAt), { addSuffix: true })}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-semibold text-gray-900">
                            {activity.petMetrics?.distance || '0'} km
                          </p>
                          <p className="text-xs text-gray-600">
                            {activity.humanMetrics?.calories || '0'} cal
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="p-8 text-center">
                  <Play className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">No activities yet</p>
                  <p className="text-xs text-gray-500 mt-1">Start tracking your first activity!</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
