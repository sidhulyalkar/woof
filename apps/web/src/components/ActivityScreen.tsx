'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Play, Pause, Square, MapPin, Timer, Footprints, Flame, Heart, TrendingUp, Share2, X } from 'lucide-react';
import { useSessionStore } from '@/store/session';
import { useActivities, useCreateActivity, useUpdateActivity, useCreatePost } from '@/lib/api/hooks';
import { useUIStore } from '@/store/ui';
import { formatDistanceToNow } from 'date-fns';

type ActivityType = 'WALK' | 'RUN' | 'PLAY' | 'HIKE';

export function ActivityScreen() {
  const router = useRouter();
  const { user, pets } = useSessionStore();
  const { showToast } = useUIStore();
  const { data: activities, isLoading } = useActivities();
  const createActivity = useCreateActivity();
  const updateActivity = useUpdateActivity();
  const createPost = useCreatePost();

  const [isTracking, setIsTracking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [selectedActivityType, setSelectedActivityType] = useState<ActivityType>('WALK');
  const [selectedPet, setSelectedPet] = useState<string | null>(pets[0]?.id || null);
  const [currentActivityId, setCurrentActivityId] = useState<string | null>(null);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [completedActivity, setCompletedActivity] = useState<any>(null);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const [currentStats, setCurrentStats] = useState({
    duration: 0,
    distance: 0,
    calories: 0,
    steps: 0,
  });

  // Real-time tracking simulation
  useEffect(() => {
    if (isTracking && !isPaused) {
      timerRef.current = setInterval(() => {
        setCurrentStats(prev => ({
          duration: prev.duration + 1,
          distance: prev.distance + (Math.random() * 0.01), // Simulate GPS
          calories: prev.calories + Math.floor(Math.random() * 2),
          steps: prev.steps + Math.floor(Math.random() * 3),
        }));
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isTracking, isPaused]);

  const handleStartActivity = async () => {
    const start = new Date();
    setStartTime(start);
    setIsTracking(true);
    setIsPaused(false);

    try {
      const activity = await createActivity.mutateAsync({
        type: selectedActivityType,
        startedAt: start.toISOString(),
        petId: selectedPet || undefined,
      });
      setCurrentActivityId(activity.id);
      showToast({ message: `${selectedActivityType} started!`, type: 'success' });
    } catch (error) {
      showToast({ message: 'Failed to start activity', type: 'error' });
      setIsTracking(false);
    }
  };

  const handlePauseActivity = () => {
    setIsPaused(!isPaused);
  };

  const handleStopActivity = async () => {
    if (!currentActivityId || !startTime) return;

    try {
      const updatedActivity = await updateActivity.mutateAsync({
        id: currentActivityId,
        data: {
          endedAt: new Date().toISOString(),
          humanMetrics: {
            steps: currentStats.steps,
            calories: currentStats.calories,
            hr_avg: 120, // Mock data
          },
          petMetrics: {
            distance: currentStats.distance.toFixed(2),
            active_time: currentStats.duration,
          },
        },
      });

      // Store completed activity data for sharing
      setCompletedActivity({
        id: currentActivityId,
        type: selectedActivityType,
        stats: currentStats,
        petId: selectedPet,
      });
      setShowShareDialog(true);

      setIsTracking(false);
      setIsPaused(false);
      setCurrentStats({ duration: 0, distance: 0, calories: 0, steps: 0 });
      setCurrentActivityId(null);
      setStartTime(null);
    } catch (error) {
      showToast({ message: 'Failed to save activity', type: 'error' });
    }
  };

  const handleShareActivity = async () => {
    if (!completedActivity || !user) return;

    const pet = pets.find(p => p.id === completedActivity.petId);
    const petName = pet ? ` with ${pet.name}` : '';
    const duration = Math.floor(completedActivity.stats.duration / 60);
    const distance = completedActivity.stats.distance.toFixed(1);

    try {
      await createPost.mutateAsync({
        content: `Just completed a ${completedActivity.type.toLowerCase()}${petName}! 🎉\n\n📍 ${distance}km in ${duration} minutes\n🔥 ${completedActivity.stats.calories} calories burned\n👟 ${completedActivity.stats.steps} steps`,
        petId: completedActivity.petId || undefined,
        activityId: completedActivity.id,
      });

      showToast({ message: 'Activity shared to feed!', type: 'success' });
      setShowShareDialog(false);
      setCompletedActivity(null);
      router.push('/');
    } catch (error) {
      showToast({ message: 'Failed to share activity', type: 'error' });
    }
  };

  const handleSkipShare = () => {
    showToast({ message: 'Activity saved!', type: 'success' });
    setShowShareDialog(false);
    setCompletedActivity(null);
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

      {/* Share Activity Dialog */}
      {showShareDialog && completedActivity && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full overflow-hidden animate-in slide-in-from-bottom-4 duration-300">
            {/* Header */}
            <div className="px-6 py-5 border-b border-gray-100 flex items-center justify-between">
              <h3 className="text-lg font-bold text-gray-900">Activity Completed! 🎉</h3>
              <button
                onClick={handleSkipShare}
                className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors"
              >
                <X className="h-5 w-5 text-gray-600" />
              </button>
            </div>

            {/* Stats Summary */}
            <div className="px-6 py-6 bg-gradient-to-br from-blue-50 to-indigo-50">
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {completedActivity.stats.distance.toFixed(1)}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">km</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {Math.floor(completedActivity.stats.duration / 60)}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">min</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {completedActivity.stats.calories}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">cal</div>
                </div>
              </div>
              <p className="text-sm text-center text-gray-700">
                Great {completedActivity.type.toLowerCase()}
                {completedActivity.petId && pets.find(p => p.id === completedActivity.petId)
                  ? ` with ${pets.find(p => p.id === completedActivity.petId)?.name}`
                  : ''}!
              </p>
            </div>

            {/* Actions */}
            <div className="px-6 py-5 space-y-3">
              <button
                onClick={handleShareActivity}
                disabled={createPost.isPending}
                className="w-full py-3.5 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl font-semibold shadow-md hover:shadow-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Share2 className="h-5 w-5" />
                Share to Feed
              </button>
              <button
                onClick={handleSkipShare}
                className="w-full py-3.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
              >
                Skip for Now
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
