'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import { FileUpload } from '@/components/ui/file-upload';
import { activitiesApi, storageApi } from '@/lib/api';
import { toast } from 'sonner';

const activityTypes = [
  { value: 'walk', label: 'Walk' },
  { value: 'run', label: 'Run' },
  { value: 'play', label: 'Play' },
  { value: 'training', label: 'Training' },
  { value: 'grooming', label: 'Grooming' },
  { value: 'vet_visit', label: 'Vet Visit' },
  { value: 'other', label: 'Other' },
];

export function ManualActivityForm({ petId, onSuccess }: { petId: string; onSuccess?: () => void }) {
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    type: 'walk',
    datetime: new Date().toISOString().slice(0, 16),
    duration: '',
    distance: '',
    calories: '',
    notes: '',
    location: '',
  });
  const [photos, setPhotos] = useState<File[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Upload photos first
      let photoUrls: string[] = [];
      if (photos.length > 0) {
        const uploads = await storageApi.uploadFiles(photos, 'activities');
        photoUrls = uploads.map((u) => u.url);
      }

      // Create activity
      await activitiesApi.logActivity({
        petId,
        type: formData.type,
        datetime: formData.datetime,
        duration: formData.duration ? parseInt(formData.duration) : undefined,
        distance: formData.distance ? parseFloat(formData.distance) : undefined,
        calories: formData.calories ? parseInt(formData.calories) : undefined,
        notes: formData.notes,
        location: formData.location,
        photos: photoUrls,
      });

      toast.success('Activity logged successfully!');
      onSuccess?.();
    } catch (error) {
      toast.error('Failed to log activity');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Card className="p-6 space-y-6">
        <h2 className="text-2xl font-bold">Log Activity</h2>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="type">Activity Type</Label>
            <Select
              value={formData.type}
              onValueChange={(value) => setFormData({ ...formData, type: value })}
            >
              <SelectTrigger>
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
            <Label htmlFor="datetime">Date & Time</Label>
            <Input
              id="datetime"
              type="datetime-local"
              value={formData.datetime}
              onChange={(e) => setFormData({ ...formData, datetime: e.target.value })}
              required
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (min)</Label>
              <Input
                id="duration"
                type="number"
                value={formData.duration}
                onChange={(e) => setFormData({ ...formData, duration: e.target.value })}
                placeholder="30"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="distance">Distance (km)</Label>
              <Input
                id="distance"
                type="number"
                step="0.1"
                value={formData.distance}
                onChange={(e) => setFormData({ ...formData, distance: e.target.value })}
                placeholder="2.5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="calories">Calories</Label>
              <Input
                id="calories"
                type="number"
                value={formData.calories}
                onChange={(e) => setFormData({ ...formData, calories: e.target.value })}
                placeholder="150"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="location">Location</Label>
            <Input
              id="location"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              placeholder="Central Park"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Notes</Label>
            <Textarea
              id="notes"
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              placeholder="Add any notes about this activity..."
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
          </div>
        </div>

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? 'Logging...' : 'Log Activity'}
        </Button>
      </Card>
    </form>
  );
}
