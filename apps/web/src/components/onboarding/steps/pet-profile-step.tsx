'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { OnboardingStepProps } from '../onboarding-types';

export function PetProfileStep({ data, onNext, onBack }: OnboardingStepProps) {
  const [name, setName] = useState(data.petName ?? '');
  const [species, setSpecies] = useState(data.petSpecies ?? 'DOG');
  const [breed, setBreed] = useState(data.petBreed ?? '');

  return (
    <Card className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold">Add your companion</h2>
        <p className="text-muted-foreground">
          A few basics help Woof personalize shared activities.
        </p>
      </div>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="pet-name">Name</Label>
          <Input
            id="pet-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            required
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="pet-species">Species</Label>
          <select
            id="pet-species"
            value={species}
            onChange={(event) => setSpecies(event.target.value)}
            className="h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="DOG">Dog</option>
            <option value="CAT">Cat</option>
            <option value="OTHER">Other</option>
          </select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="pet-breed">Breed or mix</Label>
          <Input id="pet-breed" value={breed} onChange={(event) => setBreed(event.target.value)} />
        </div>
      </div>

      <div className="flex gap-3">
        {onBack && (
          <Button type="button" variant="outline" onClick={onBack}>
            Back
          </Button>
        )}
        <Button
          type="button"
          className="flex-1"
          disabled={!name.trim()}
          onClick={() =>
            onNext({ petName: name.trim(), petSpecies: species, petBreed: breed.trim() })
          }
        >
          Continue
        </Button>
      </div>
    </Card>
  );
}
