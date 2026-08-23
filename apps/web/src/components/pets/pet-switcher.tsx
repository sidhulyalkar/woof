'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { PawPrint } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { setActivePetId } from '@/lib/active-pet';
import { petsApi } from '@/lib/api/pets';

export function PetSwitcher({
  currentPetId,
  label = 'Today is for',
  onChange,
}: {
  currentPetId: string;
  label?: string;
  onChange?: (petId: string) => void;
}) {
  const queryClient = useQueryClient();
  const pets = useQuery({
    queryKey: ['pets', 'mine'],
    queryFn: () => petsApi.getMine(),
    staleTime: 60_000,
    retry: false,
  });

  const ownedPets = pets.data?.pets ?? [];
  if (pets.isError || ownedPets.length <= 1) return null;

  const choosePet = async (petId: string) => {
    if (petId === currentPetId) return;
    setActivePetId(petId);
    onChange?.(petId);
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ['adventure', 'me'] }),
      queryClient.invalidateQueries({ queryKey: ['concierge', 'today'] }),
      queryClient.invalidateQueries({ queryKey: ['activities'] }),
      queryClient.invalidateQueries({ queryKey: ['story'] }),
      queryClient.invalidateQueries({ queryKey: ['recommendations'] }),
      queryClient.invalidateQueries({ queryKey: ['discovery', 'nearby'] }),
    ]);
  };

  return (
    <div className="mt-4 border-t border-border/60 pt-3">
      <p className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </p>
      <div className="flex gap-2 overflow-x-auto pb-1" role="group" aria-label="Choose active dog">
        {ownedPets.map((pet) => {
          const active = pet.id === currentPetId;
          return (
            <button
              key={pet.id}
              type="button"
              aria-pressed={active}
              onClick={() => void choosePet(pet.id)}
              className={`flex shrink-0 items-center gap-2 rounded-full border px-3 py-2 text-sm font-semibold transition-colors ${
                active
                  ? 'border-primary/30 bg-primary/10 text-primary'
                  : 'border-border/70 bg-background/55 text-muted-foreground hover:border-primary/30 hover:text-foreground'
              }`}
            >
              <Avatar className="h-6 w-6 border-0">
                <AvatarImage src={pet.avatarUrl ?? undefined} alt="" />
                <AvatarFallback className="bg-primary/10 text-primary">
                  <PawPrint className="h-3.5 w-3.5" aria-hidden="true" />
                </AvatarFallback>
              </Avatar>
              {pet.name}
            </button>
          );
        })}
      </div>
    </div>
  );
}
