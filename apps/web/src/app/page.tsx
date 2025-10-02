'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Heart, Footprints, TrendingUp, Camera } from 'lucide-react';
import { petsApi, activitiesApi } from '@/lib/api';
import { PetCard } from '@/components/dashboard/pet-card';
import { WeeklyProgress } from '@/components/dashboard/weekly-progress';
import { HappinessScore } from '@/components/dashboard/happiness-score';

export default function HomePage() {
  const [happinessScore] = useState(87);

  // Fetch user's pets
  const { data: pets, isLoading: petsLoading } = useQuery({
    queryKey: ['pets'],
    queryFn: petsApi.getAllPets,
  });

  // Fetch activities
  const { data: activities, isLoading: activitiesLoading } = useQuery({
    queryKey: ['activities'],
    queryFn: activitiesApi.getAllActivities,
  });

  const isLoading = petsLoading || activitiesLoading;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="gradient-primary rounded-2xl p-6 text-white relative overflow-hidden">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-64 h-64 bg-accent rounded-full blur-3xl" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent-600 rounded-full blur-3xl" />
          </div>

          <div className="relative z-10">
            <h1 className="text-3xl font-heading font-bold mb-2">Welcome back! 👋</h1>
            <p className="text-accent-100">Ready for another adventure with your furry friends?</p>

            <HappinessScore score={happinessScore} />
          </div>
        </div>

        {/* Pet Cards */}
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2].map((i) => (
              <div key={i} className="h-48 bg-surface/50 rounded-lg animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pets?.map((pet: any) => (
              <PetCard key={pet.id} pet={pet} />
            ))}
          </div>
        )}

        {/* Weekly Progress */}
        <WeeklyProgress activities={activities || []} />
      </div>
    </div>
  );
}
