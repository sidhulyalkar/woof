'use client';

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { ArrowLeft, Loader2, PawPrint, Sparkles } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { FormEvent, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { setActivePetId } from '@/lib/active-pet';
import { petsApi } from '@/lib/api/pets';

export default function NewPetPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [name, setName] = useState('');
  const [breed, setBreed] = useState('');
  const [birthdate, setBirthdate] = useState('');
  const [sex, setSex] = useState<'MALE' | 'FEMALE' | 'UNKNOWN'>('UNKNOWN');

  const createDog = useMutation({
    mutationFn: () =>
      petsApi.createDog({
        name: name.trim(),
        species: 'DOG',
        ...(breed.trim() ? { breed: breed.trim() } : {}),
        ...(birthdate ? { birthdate } : {}),
        sex,
      }),
    onSuccess: async (pet) => {
      setActivePetId(pet.id);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['pets', 'mine'] }),
        queryClient.invalidateQueries({ queryKey: ['adventure', 'me'] }),
        queryClient.invalidateQueries({ queryKey: ['concierge', 'today'] }),
      ]);
      router.replace('/');
    },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!name.trim() || createDog.isPending) return;
    createDog.mutate();
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <Button variant="ghost" size="icon" asChild className="rounded-xl">
            <Link href="/profile" aria-label="Back to profile">
              <ArrowLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div>
            <p className="eyebrow">Start dogOS</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Add your dog</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl px-4 py-6">
        <Card className="rounded-3xl border-primary/15 bg-gradient-to-br from-primary/[0.08] via-card/95 to-secondary/[0.04] p-5 sm:p-6">
          <div className="flex items-start gap-3">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <PawPrint className="h-5 w-5" aria-hidden="true" />
            </span>
            <div>
              <p className="eyebrow">One minute setup</p>
              <h2 className="mt-1 text-xl font-bold">Who are we caring for?</h2>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                Start with only the basics. You can enrich the profile later as Woof earns more
                context from real activities and your explicit feedback.
              </p>
            </div>
          </div>

          <form className="mt-6 space-y-5" onSubmit={submit}>
            <div className="space-y-2">
              <Label htmlFor="dog-name">Name</Label>
              <Input
                id="dog-name"
                autoFocus
                autoComplete="off"
                maxLength={80}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="e.g. Shasta"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="dog-breed">Breed or mix <span className="text-muted-foreground">(optional)</span></Label>
              <Input
                id="dog-breed"
                autoComplete="off"
                maxLength={120}
                value={breed}
                onChange={(event) => setBreed(event.target.value)}
                placeholder="e.g. Siberian Husky"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="dog-birthdate">Birthday <span className="text-muted-foreground">(optional)</span></Label>
                <Input
                  id="dog-birthdate"
                  type="date"
                  value={birthdate}
                  max={new Date().toISOString().slice(0, 10)}
                  onChange={(event) => setBirthdate(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="dog-sex">Sex <span className="text-muted-foreground">(optional)</span></Label>
                <select
                  id="dog-sex"
                  value={sex}
                  onChange={(event) => setSex(event.target.value as typeof sex)}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value="UNKNOWN">Prefer not to say</option>
                  <option value="FEMALE">Female</option>
                  <option value="MALE">Male</option>
                </select>
              </div>
            </div>

            <div className="rounded-2xl border border-border/60 bg-background/55 p-4 text-xs leading-relaxed text-muted-foreground">
              <div className="flex items-start gap-2">
                <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
                <p>
                  Woof does not infer medical status from this profile. Care guidance stays
                  non-diagnostic, and you remain in control of every persistent change.
                </p>
              </div>
            </div>

            {createDog.isError && (
              <p className="text-sm text-destructive" role="alert">
                We couldn&apos;t create this dog profile. Nothing was saved. Please try again.
              </p>
            )}

            <Button className="w-full" size="lg" type="submit" disabled={!name.trim() || createDog.isPending}>
              {createDog.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              )}
              Meet {name.trim() || 'your dog'}
            </Button>
          </form>
        </Card>
      </main>
    </div>
  );
}
