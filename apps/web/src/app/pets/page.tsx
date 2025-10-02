'use client';

import { useQuery } from '@tanstack/react-query';
import { Plus, Footprints } from 'lucide-react';
import { petsApi } from '@/lib/api';
import { PetCard } from '@/components/dashboard/pet-card';

export default function PetsPage() {
  const { data: pets, isLoading } = useQuery({
    queryKey: ['pets'],
    queryFn: petsApi.getAllPets,
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-heading font-bold text-gray-100">My Pets</h1>
        <button className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-accent to-accent-600 text-white rounded-lg hover:shadow-lg hover:shadow-accent/20 transition-all">
          <Plus className="w-4 h-4" />
          <span>Add Pet</span>
        </button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-48 bg-surface/50 rounded-lg animate-pulse" />
          ))}
        </div>
      ) : pets && pets.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {pets.map((pet: any) => (
            <PetCard key={pet.id} pet={pet} />
          ))}
        </div>
      ) : (
        <div className="bg-surface/50 backdrop-blur-sm border border-primary/20 rounded-lg p-12 text-center">
          <Footprints className="w-12 h-12 text-gray-500 mx-auto mb-4" />
          <h3 className="font-heading font-semibold text-gray-300 mb-2">No pets yet</h3>
          <p className="text-gray-400 mb-4">Add your first pet to get started!</p>
          <button className="px-6 py-2 bg-gradient-to-r from-accent to-accent-600 text-white rounded-lg hover:shadow-lg hover:shadow-accent/20 transition-all">
            Add Pet
          </button>
        </div>
      )}
    </div>
  );
}
