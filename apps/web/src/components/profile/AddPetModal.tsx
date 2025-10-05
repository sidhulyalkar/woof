'use client';

import { useState } from 'react';
import { X, Loader2 } from 'lucide-react';
import { useCreatePet } from '@/lib/api/hooks';
import { useUIStore } from '@/store/ui';

interface AddPetModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AddPetModal({ isOpen, onClose }: AddPetModalProps) {
  const { showToast } = useUIStore();
  const createPet = useCreatePet();

  const [formData, setFormData] = useState({
    name: '',
    species: 'DOG' as 'DOG' | 'CAT' | 'OTHER',
    breed: '',
    age: '',
    weight: '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name.trim()) {
      showToast({ message: 'Please enter a pet name', type: 'warning' });
      return;
    }

    try {
      await createPet.mutateAsync({
        name: formData.name.trim(),
        species: formData.species,
        breed: formData.breed.trim() || undefined,
        age: formData.age ? parseInt(formData.age) : undefined,
        weight: formData.weight ? parseFloat(formData.weight) : undefined,
      });

      showToast({ message: `${formData.name} added successfully!`, type: 'success' });
      setFormData({ name: '', species: 'DOG', breed: '', age: '', weight: '' });
      onClose();
    } catch (error) {
      showToast({ message: 'Failed to add pet', type: 'error' });
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-3xl shadow-2xl max-w-md w-full overflow-hidden animate-in slide-in-from-bottom-4 duration-300">
        {/* Header */}
        <div className="px-6 py-5 border-b border-gray-100 flex items-center justify-between">
          <h3 className="text-lg font-bold text-gray-900">Add New Pet</h3>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors"
          >
            <X className="h-5 w-5 text-gray-600" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 py-6 space-y-4">
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Pet Name *
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g. Buddy"
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              autoFocus
            />
          </div>

          {/* Species */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Species *
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['DOG', 'CAT', 'OTHER'] as const).map((species) => (
                <button
                  key={species}
                  type="button"
                  onClick={() => setFormData({ ...formData, species })}
                  className={`py-3 px-4 rounded-xl border-2 font-medium text-sm transition-all ${
                    formData.species === species
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {species === 'DOG' ? '🐕 Dog' : species === 'CAT' ? '🐱 Cat' : '🐾 Other'}
                </button>
              ))}
            </div>
          </div>

          {/* Breed */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Breed (Optional)
            </label>
            <input
              type="text"
              value={formData.breed}
              onChange={(e) => setFormData({ ...formData, breed: e.target.value })}
              placeholder="e.g. Golden Retriever"
              className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
          </div>

          {/* Age and Weight */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Age (Optional)
              </label>
              <input
                type="number"
                value={formData.age}
                onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                placeholder="Years"
                min="0"
                max="30"
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Weight (Optional)
              </label>
              <input
                type="number"
                value={formData.weight}
                onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
                placeholder="kg"
                min="0"
                step="0.1"
                className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-3.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-xl font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createPet.isPending || !formData.name.trim()}
              className="flex-1 py-3.5 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-xl font-semibold shadow-md hover:shadow-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {createPet.isPending ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Adding...
                </>
              ) : (
                'Add Pet'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
