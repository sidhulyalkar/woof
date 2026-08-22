'use client';

import type React from 'react';
import { useState } from 'react';
import { AtSign, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

export interface OwnerInfoData {
  handle: string;
  email: string;
  password: string;
  bio: string;
}

interface OwnerInfoStepProps {
  onComplete: (data: OwnerInfoData) => void;
  initialData?: OwnerInfoData | null;
}

export function OwnerInfoStep({ onComplete, initialData }: OwnerInfoStepProps) {
  const [formData, setFormData] = useState<OwnerInfoData>({
    handle: initialData?.handle || '',
    email: initialData?.email || '',
    password: initialData?.password || '',
    bio: initialData?.bio || '',
  });

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onComplete({
      ...formData,
      handle: formData.handle.trim().toLowerCase().replace(/\s+/g, '_'),
    });
  };

  const isValid =
    formData.handle.trim().length >= 3 &&
    formData.email.trim().length > 0 &&
    formData.password.length >= 8 &&
    formData.bio.length <= 500;

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <p className="eyebrow">Owner account</p>
        <h1 className="text-3xl font-bold tracking-tight text-balance">
          Create the human side of the profile
        </h1>
        <p className="text-sm leading-relaxed text-muted-foreground text-pretty">
          Woof starts with only the account data it can actually use. Precise location, schedules,
          and other sensitive context should be requested later and only when a feature needs them.
        </p>
      </div>

      <Card className="surface-soft flex items-start gap-3 rounded-2xl p-4">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
          <ShieldCheck className="h-5 w-5" aria-hidden="true" />
        </span>
        <div>
          <p className="text-sm font-semibold">Data minimization by default</p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            We do not ask for your age, home location, or route history just to create an account.
          </p>
        </div>
      </Card>

      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="handle">Public handle</Label>
          <div className="relative">
            <AtSign
              className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              id="handle"
              autoComplete="username"
              placeholder="trailpaws"
              value={formData.handle}
              onChange={(event) => setFormData({ ...formData, handle: event.target.value })}
              className="pl-9"
              minLength={3}
              maxLength={30}
              required
            />
          </div>
          <p className="text-xs text-muted-foreground">
            3–30 characters. Spaces are converted to underscores.
          </p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            placeholder="you@example.com"
            value={formData.email}
            onChange={(event) => setFormData({ ...formData, email: event.target.value })}
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            autoComplete="new-password"
            placeholder="Create a secure password"
            value={formData.password}
            onChange={(event) => setFormData({ ...formData, password: event.target.value })}
            required
            minLength={8}
          />
          <p className="text-xs text-muted-foreground">At least 8 characters.</p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="bio">
            Short bio <span className="text-muted-foreground">(optional)</span>
          </Label>
          <Textarea
            id="bio"
            placeholder="What kinds of dog-friendly activities are you usually up for?"
            value={formData.bio}
            onChange={(event) =>
              setFormData({ ...formData, bio: event.target.value.slice(0, 500) })
            }
            rows={4}
            className="resize-none"
          />
          <p className="text-right text-xs text-muted-foreground">{formData.bio.length}/500</p>
        </div>
      </div>

      <Button type="submit" size="lg" className="w-full" disabled={!isValid}>
        Continue to pet profile
      </Button>
    </form>
  );
}
