'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';
import type { OnboardingStepProps } from '../onboarding-types';

export function UserProfileStep({ data, onNext }: OnboardingStepProps) {
  const [formData, setFormData] = useState({
    name: data.name ?? '',
    bio: data.bio ?? '',
    location: data.location ?? '',
  });

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onNext(formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <Card className="p-6 space-y-6">
        <div className="space-y-2">
          <h2 className="text-2xl font-bold">Tell us about yourself</h2>
          <p className="text-muted-foreground">Help other pet owners get to know you</p>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(event) => setFormData({ ...formData, name: event.target.value })}
              placeholder="Your name"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="bio">Bio</Label>
            <Textarea
              id="bio"
              value={formData.bio}
              onChange={(event) => setFormData({ ...formData, bio: event.target.value })}
              placeholder="Tell us a bit about yourself..."
              rows={4}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="location">Location</Label>
            <Input
              id="location"
              value={formData.location}
              onChange={(event) => setFormData({ ...formData, location: event.target.value })}
              placeholder="City, State"
            />
          </div>
        </div>

        <Button type="submit" className="w-full">
          Continue
        </Button>
      </Card>
    </form>
  );
}
