'use client';

import { useState } from 'react';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { FileUpload } from '@/components/ui/file-upload';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { storageApi } from '@/lib/api';
import { apiClient } from '@/lib/api/client';

const activityTypes = [
  { value: 'WALK', label: 'Walk' },
  { value: 'RUN', label: 'Run' },
  { value: 'PLAY', label: 'Play' },
  { value: 'HIKE', label: 'Hike' },
  { value: 'TRAINING', label: 'Training' },
  { value: 'GROOMING', label: 'Grooming' },
  { value: 'VET_VISIT', label: 'Vet visit' },
  { value: 'OTHER', label: 'Other' },
];

type ActivityFormState = {
  type: string;
  datetime: string;
  duration: string;
  distance: string;
  calories: string;
  notes: string;
  location: string;
};

export function ManualActivityForm({
  petId,
  onSuccess,
}: {
  petId: string;
  onSuccess?: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState<ActivityFormState>({
    type: 'WALK',
    datetime: new Date().toISOString().slice(0, 16),
    duration: '',
    distance: '',
    calories: '',
    notes: '',
    location: '',
  });
  const [photos, setPhotos] = useState<File[]>([]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);

    try {
      let photoUrls: string[] = [];
      if (photos.length > 0) {
        try {
          const uploads = await storageApi.uploadFiles(photos, 'activities');
          photoUrls = uploads.map((upload) => upload.url);
        } catch {
          toast.warning('Activity will be saved without photos');
        }
      }

      const startedAt = new Date(formData.datetime);
      const durationMinutes = formData.duration
        ? Math.max(0, Number.parseInt(formData.duration, 10))
        : undefined;
      const endedAt = durationMinutes
        ? new Date(startedAt.getTime() + durationMinutes * 60_000)
        : undefined;
      const distanceKm = formData.distance
        ? Math.max(0, Number.parseFloat(formData.distance))
        : undefined;
      const calories = formData.calories
        ? Math.max(0, Number.parseInt(formData.calories, 10))
        : undefined;

      await apiClient.post('/activities', {
        petId,
        type: formData.type,
        startedAt: startedAt.toISOString(),
        ...(endedAt ? { endedAt: endedAt.toISOString() } : {}),
        humanMetrics: {
          ...(calories !== undefined ? { calories } : {}),
        },
        petMetrics: {
          ...(distanceKm !== undefined ? { distanceKm } : {}),
          ...(durationMinutes !== undefined ? { activeMinutes: durationMinutes } : {}),
        },
        jointMetrics: {
          contextVersion: 'activity-context-v1',
          ...(formData.notes.trim() ? { notes: formData.notes.trim() } : {}),
          ...(formData.location.trim()
            ? { locationLabel: formData.location.trim() }
            : {}),
          ...(photoUrls.length > 0 ? { mediaUrls: photoUrls } : {}),
          entryMethod: 'manual',
        },
      });

      toast.success('Activity saved. Woof can use it to learn your shared routine.');
      setFormData((current) => ({
        ...current,
        duration: '',
        distance: '',
        calories: '',
        notes: '',
        location: '',
      }));
      setPhotos([]);
      onSuccess?.();
    } catch (error) {
      toast.error('Activity could not be saved');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Card className="space-y-6 p-6">
        <div>
          <p className="eyebrow">Shared experience</p>
          <h2 className="mt-1 text-2xl font-bold">Log an activity</h2>
          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
            Record enough context to remember the moment and help Woof learn patterns over
            time. This is relationship context, not a medical record.
          </p>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="type">Activity type</Label>
            <Select
              value={formData.type}
              onValueChange={(value) =>
                setFormData((current) => ({ ...current, type: value }))
              }
            >
              <SelectTrigger id="type">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {activityTypes.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="datetime">Date &amp; time</Label>
            <Input
              id="datetime"
              type="datetime-local"
              value={formData.datetime}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  datetime: event.target.value,
                }))
              }
              required
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (min)</Label>
              <Input
                id="duration"
                type="number"
                min="0"
                value={formData.duration}
                onChange={(event) =>
                  setFormData((current) => ({
                    ...current,
                    duration: event.target.value,
                  }))
                }
                placeholder="30"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="distance">Distance (km)</Label>
              <Input
                id="distance"
                type="number"
                min="0"
                step="0.1"
                value={formData.distance}
                onChange={(event) =>
                  setFormData((current) => ({
                    ...current,
                    distance: event.target.value,
                  }))
                }
                placeholder="2.5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="calories">Calories</Label>
              <Input
                id="calories"
                type="number"
                min="0"
                value={formData.calories}
                onChange={(event) =>
                  setFormData((current) => ({
                    ...current,
                    calories: event.target.value,
                  }))
                }
                placeholder="150"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="location">Place label</Label>
            <Input
              id="location"
              value={formData.location}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  location: event.target.value,
                }))
              }
              placeholder="Neighborhood trail"
            />
            <p className="text-xs text-muted-foreground">
              A label is enough. Woof does not need a precise route for manual entries.
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">What did you notice?</Label>
            <Textarea
              id="notes"
              value={formData.notes}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  notes: event.target.value,
                }))
              }
              placeholder="Calm at the start, excited around other dogs, settled quickly afterward…"
              rows={3}
            />
          </div>

          <div className="space-y-2">
            <Label>Photos</Label>
            <FileUpload
              onUpload={(files) => setPhotos(files)}
              multiple
              accept="image/*"
              value={photos}
            />
            <p className="text-xs text-muted-foreground">
              Optional. A storage outage will not block the activity itself.
            </p>
          </div>
        </div>

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? 'Saving…' : 'Save shared activity'}
        </Button>
      </Card>
    </form>
  );
}
