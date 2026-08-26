'use client';

import type React from 'react';
import { useState } from 'react';
import { PawPrint, Upload, X } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export interface PetInfoData {
  name: string;
  species: string;
  breed: string;
  birthdate: string;
  photoFile: File | null;
  photoPreviewUrl: string;
}

interface PetInfoStepProps {
  onComplete: (data: PetInfoData) => void;
  initialData?: PetInfoData | null;
  isLoading?: boolean;
}

export function PetInfoStep({ onComplete, initialData, isLoading = false }: PetInfoStepProps) {
  const [formData, setFormData] = useState<PetInfoData>({
    name: initialData?.name || '',
    species: initialData?.species || 'DOG',
    breed: initialData?.breed || '',
    birthdate: initialData?.birthdate || '',
    photoFile: initialData?.photoFile || null,
    photoPreviewUrl: initialData?.photoPreviewUrl || '',
  });

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onComplete({ ...formData, name: formData.name.trim(), breed: formData.breed.trim() });
  };

  const handlePhoto = (file?: File) => {
    if (!file) return;

    if (formData.photoPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(formData.photoPreviewUrl);
    }

    setFormData((current) => ({
      ...current,
      photoFile: file,
      photoPreviewUrl: URL.createObjectURL(file),
    }));
  };

  const removePhoto = () => {
    if (formData.photoPreviewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(formData.photoPreviewUrl);
    }
    setFormData((current) => ({ ...current, photoFile: null, photoPreviewUrl: '' }));
  };

  const isValid = Boolean(formData.name.trim() && formData.species && formData.birthdate);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <p className="eyebrow">Your pet</p>
        <h1 className="text-3xl font-bold tracking-tight text-balance">
          A few facts, then we can do something together
        </h1>
        <p className="text-sm leading-relaxed text-muted-foreground text-pretty">
          Woof only needs durable basics here. Preferences, comfort, and what actually works are
          better learned from real Adventures than from asking you to describe your pet all at once.
        </p>
      </div>

      <Card className="glass rounded-2xl p-5">
        <div className="flex items-center gap-4">
          <Avatar className="h-20 w-20 border-2 border-border">
            <AvatarImage src={formData.photoPreviewUrl || '/placeholder.svg'} alt="Pet preview" />
            <AvatarFallback className="bg-secondary/10 text-secondary">
              <PawPrint className="h-8 w-8" aria-hidden="true" />
            </AvatarFallback>
          </Avatar>
          <div className="min-w-0 flex-1">
            <Label
              htmlFor="pet-photo"
              className="inline-flex min-h-11 cursor-pointer items-center gap-2 rounded-xl border border-border px-3 py-2 text-sm font-semibold hover:border-primary/40 hover:bg-primary/5"
            >
              <Upload className="h-4 w-4" aria-hidden="true" />
              {formData.photoFile ? 'Replace photo' : 'Choose a photo'}
            </Label>
            <Input
              id="pet-photo"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(event) => handlePhoto(event.target.files?.[0])}
              disabled={isLoading}
            />
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              Optional. We create your pet first, then attach the photo. A failed upload can never
              block your first Adventure.
            </p>
            {formData.photoFile && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="mt-1 h-auto min-h-0 px-0 text-muted-foreground hover:text-destructive"
                onClick={removePhoto}
                disabled={isLoading}
              >
                <X className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
                Remove
              </Button>
            )}
          </div>
        </div>
      </Card>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="pet-name">Pet name</Label>
          <Input
            id="pet-name"
            autoComplete="off"
            placeholder="Shasta"
            value={formData.name}
            onChange={(event) => setFormData({ ...formData, name: event.target.value })}
            required
            disabled={isLoading}
          />
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="species">Species</Label>
            <Select
              value={formData.species}
              onValueChange={(value) => setFormData({ ...formData, species: value })}
              disabled={isLoading}
            >
              <SelectTrigger id="species">
                <SelectValue placeholder="Select species" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="DOG">Dog</SelectItem>
                <SelectItem value="CAT">Cat</SelectItem>
                <SelectItem value="OTHER">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="birthdate">Birthday or best estimate</Label>
            <Input
              id="birthdate"
              type="date"
              value={formData.birthdate}
              max={new Date().toISOString().split('T')[0]}
              onChange={(event) => setFormData({ ...formData, birthdate: event.target.value })}
              required
              disabled={isLoading}
            />
            <p className="text-xs text-muted-foreground">
              Life stage can change which activities are a sensible starting point.
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="breed">
            Breed or mix <span className="text-muted-foreground">(optional)</span>
          </Label>
          <Input
            id="breed"
            placeholder="Siberian Husky"
            value={formData.breed}
            onChange={(event) => setFormData({ ...formData, breed: event.target.value })}
            disabled={isLoading}
          />
          <p className="text-xs text-muted-foreground">
            Supporting context only. Woof learns this individual from their own outcomes rather than
            treating breed as destiny.
          </p>
        </div>
      </div>

      <Button type="submit" size="lg" className="w-full" disabled={!isValid || isLoading}>
        {isLoading ? 'Creating your pair…' : 'Continue to First Adventure'}
      </Button>
    </form>
  );
}
