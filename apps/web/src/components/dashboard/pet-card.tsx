'use client';

import { Footprints } from 'lucide-react';
import { formatDistance } from '@/lib/utils';
import { getInitials } from '@/lib/utils';

interface PetCardProps {
  pet: {
    id: string;
    name: string;
    species: string;
    breed?: string;
    age?: number;
    avatarUrl?: string;
    energyLevel?: number;
  };
}

export function PetCard({ pet }: PetCardProps) {
  const energyLevel = pet.energyLevel || 75;
  const todaySteps = 8432; // TODO: Get from activities
  const todayDistance = 6.2; // TODO: Get from activities
  const todayCalories = 342; // TODO: Get from activities

  const getEnergyBadge = (level: number) => {
    if (level > 70) return { text: 'Energetic', color: 'bg-green-500/20 text-green-400 border-green-500/30' };
    if (level > 40) return { text: 'Calm', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' };
    return { text: 'Resting', color: 'bg-gray-500/20 text-gray-400 border-gray-500/30' };
  };

  const badge = getEnergyBadge(energyLevel);

  return (
    <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg overflow-hidden hover:border-accent/30 hover:shadow-lg hover:shadow-accent/5 transition-all">
      <div className="p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-accent to-accent-600 flex items-center justify-center text-white font-medium">
              {pet.avatarUrl ? (
                <img src={pet.avatarUrl} alt={pet.name} className="w-full h-full rounded-full object-cover" />
              ) : (
                <span>{getInitials(pet.name)}</span>
              )}
            </div>
            <div>
              <h3 className="font-heading font-semibold text-gray-100">{pet.name}</h3>
              <p className="text-sm text-gray-400">
                {pet.breed || pet.species}
                {pet.age && ` • ${pet.age} years`}
              </p>
            </div>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-medium border ${badge.color}`}>
            {badge.text}
          </span>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Today's Activity</span>
            <span className="font-medium text-gray-200">{todaySteps.toLocaleString()} steps</span>
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-primary/30 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent to-accent-600 rounded-full transition-all duration-500"
              style={{ width: `${Math.min((todaySteps / 10000) * 100, 100)}%` }}
            />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-3 gap-2">
            <div className="bg-primary/30 rounded-lg p-2 text-center border border-primary/20">
              <p className="text-xs text-gray-400">Distance</p>
              <p className="font-semibold text-gray-100">{todayDistance} km</p>
            </div>
            <div className="bg-primary/30 rounded-lg p-2 text-center border border-primary/20">
              <p className="text-xs text-gray-400">Calories</p>
              <p className="font-semibold text-gray-100">{todayCalories}</p>
            </div>
            <div className="bg-primary/30 rounded-lg p-2 text-center border border-primary/20">
              <p className="text-xs text-gray-400">Energy</p>
              <p className="font-semibold text-gray-100">{energyLevel}%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
