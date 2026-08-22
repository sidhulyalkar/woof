'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface OnboardingStepProps {
  data: Record<string, any>;
  onNext: (data: Record<string, any>) => void;
  onBack?: () => void;
}

export function PermissionsStep({ data, onNext, onBack }: OnboardingStepProps) {
  const [notifications, setNotifications] = useState(data.allowNotifications === true);
  const [location, setLocation] = useState(data.allowLocation === true);

  return (
    <Card className="space-y-6 p-6">
      <div>
        <h2 className="text-2xl font-bold">Optional permissions</h2>
        <p className="text-muted-foreground">
          Choose what you may want to enable later. Woof will still ask through your browser or
          device before accessing anything.
        </p>
      </div>

      <div className="space-y-3">
        <label className="flex items-start gap-3 rounded-xl border p-4">
          <input
            type="checkbox"
            checked={notifications}
            onChange={(event) => setNotifications(event.target.checked)}
            className="mt-1 h-4 w-4"
          />
          <span>
            <span className="block font-medium">Helpful reminders</span>
            <span className="text-sm text-muted-foreground">
              Allow Woof to offer notification setup later.
            </span>
          </span>
        </label>
        <label className="flex items-start gap-3 rounded-xl border p-4">
          <input
            type="checkbox"
            checked={location}
            onChange={(event) => setLocation(event.target.checked)}
            className="mt-1 h-4 w-4"
          />
          <span>
            <span className="block font-medium">Nearby experiences</span>
            <span className="text-sm text-muted-foreground">
              Allow Woof to offer location access when a map feature needs it.
            </span>
          </span>
        </label>
      </div>

      <p className="text-xs text-muted-foreground">
        These choices do not grant OS permissions by themselves and can be changed later.
      </p>

      <div className="flex gap-3">
        {onBack && (
          <Button variant="outline" onClick={onBack}>
            Back
          </Button>
        )}
        <Button
          className="flex-1"
          onClick={() => onNext({ allowNotifications: notifications, allowLocation: location })}
        >
          Finish setup
        </Button>
      </div>
    </Card>
  );
}
