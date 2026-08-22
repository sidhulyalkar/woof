'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Label } from '@/components/ui/label';

interface OnboardingStepProps {
  data: Record<string, any>;
  onNext: (data: Record<string, any>) => void;
  onBack?: () => void;
}

const options = [
  { id: 'explore', label: 'Explore new places' },
  { id: 'enrich', label: 'Scent and enrichment games' },
  { id: 'learn', label: 'Short skill sessions' },
  { id: 'recover', label: 'Recovery and decompression' },
];

export function PreferencesStep({ data, onNext, onBack }: OnboardingStepProps) {
  const initial = Array.isArray(data.activityPreferences)
    ? data.activityPreferences.filter((value): value is string => typeof value === 'string')
    : [];
  const [selected, setSelected] = useState<string[]>(initial);

  const toggle = (id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id]
    );
  };

  return (
    <Card className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold">What fits your life?</h2>
        <p className="text-muted-foreground">
          Pick any activities that sound useful. Woof can adapt later from real outcomes.
        </p>
      </div>

      <fieldset className="space-y-3">
        <Label asChild>
          <legend>Activity preferences</legend>
        </Label>
        {options.map((option) => {
          const checked = selected.includes(option.id);
          return (
            <label
              key={option.id}
              className="flex cursor-pointer items-center gap-3 rounded-xl border p-3"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggle(option.id)}
                className="h-4 w-4"
              />
              <span>{option.label}</span>
            </label>
          );
        })}
      </fieldset>

      <div className="flex gap-3">
        {onBack && (
          <Button variant="outline" onClick={onBack}>
            Back
          </Button>
        )}
        <Button className="flex-1" onClick={() => onNext({ activityPreferences: selected })}>
          Continue
        </Button>
      </div>
    </Card>
  );
}
