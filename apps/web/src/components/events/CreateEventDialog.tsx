'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

interface CreateEventDialogProps {
  open: boolean;
  onClose: () => void;
}

export function CreateEventDialog({ open, onClose }: CreateEventDialogProps) {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    type: 'group_walk',
    startTime: '',
    endTime: '',
    locationName: '',
    lat: 0,
    lng: 0,
    maxAttendees: '',
    tags: '',
  });

  const queryClient = useQueryClient();

  const createEventMutation = useMutation({
    mutationFn: (data: any) => apiClient.post('/events', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events'] });
      toast.success('Event created successfully!');
      onClose();
      resetForm();
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to create event');
    },
  });

  const resetForm = () => {
    setFormData({
      title: '',
      description: '',
      type: 'group_walk',
      startTime: '',
      endTime: '',
      locationName: '',
      lat: 0,
      lng: 0,
      maxAttendees: '',
      tags: '',
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const payload = {
      title: formData.title,
      description: formData.description,
      type: formData.type,
      startTime: formData.startTime,
      endTime: formData.endTime,
      locationName: formData.locationName,
      lat: formData.lat || 0,
      lng: formData.lng || 0,
      maxAttendees: formData.maxAttendees ? parseInt(formData.maxAttendees) : undefined,
      tags: formData.tags ? formData.tags.split(',').map(t => t.trim()).filter(Boolean) : [],
    };

    createEventMutation.mutate(payload);
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create New Event</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="title">Event Title *</Label>
            <Input
              id="title"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              placeholder="Sunday Morning Dog Walk"
              required
            />
          </div>

          <div>
            <Label htmlFor="type">Event Type *</Label>
            <Select
              value={formData.type}
              onValueChange={(value) => setFormData({ ...formData, type: value })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="group_walk">Group Walk</SelectItem>
                <SelectItem value="park_meetup">Park Meetup</SelectItem>
                <SelectItem value="training_session">Training Session</SelectItem>
                <SelectItem value="social_event">Social Event</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="description">Description *</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Let's enjoy a relaxing morning walk around the park..."
              rows={4}
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="startTime">Start Time *</Label>
              <Input
                id="startTime"
                type="datetime-local"
                value={formData.startTime}
                onChange={(e) => setFormData({ ...formData, startTime: e.target.value })}
                required
              />
            </div>
            <div>
              <Label htmlFor="endTime">End Time *</Label>
              <Input
                id="endTime"
                type="datetime-local"
                value={formData.endTime}
                onChange={(e) => setFormData({ ...formData, endTime: e.target.value })}
                required
              />
            </div>
          </div>

          <div>
            <Label htmlFor="locationName">Location *</Label>
            <Input
              id="locationName"
              value={formData.locationName}
              onChange={(e) => setFormData({ ...formData, locationName: e.target.value })}
              placeholder="Central Park, Main Entrance"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="lat">Latitude (optional)</Label>
              <Input
                id="lat"
                type="number"
                step="any"
                value={formData.lat}
                onChange={(e) => setFormData({ ...formData, lat: parseFloat(e.target.value) || 0 })}
                placeholder="40.785091"
              />
            </div>
            <div>
              <Label htmlFor="lng">Longitude (optional)</Label>
              <Input
                id="lng"
                type="number"
                step="any"
                value={formData.lng}
                onChange={(e) => setFormData({ ...formData, lng: parseFloat(e.target.value) || 0 })}
                placeholder="-73.968285"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="maxAttendees">Max Attendees (optional)</Label>
            <Input
              id="maxAttendees"
              type="number"
              value={formData.maxAttendees}
              onChange={(e) => setFormData({ ...formData, maxAttendees: e.target.value })}
              placeholder="10"
            />
          </div>

          <div>
            <Label htmlFor="tags">Tags (comma-separated, optional)</Label>
            <Input
              id="tags"
              value={formData.tags}
              onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
              placeholder="beginner-friendly, small-dogs, off-leash"
            />
          </div>

          <div className="flex gap-2 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => { onClose(); resetForm(); }}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createEventMutation.isPending}
              className="flex-1"
            >
              {createEventMutation.isPending ? 'Creating...' : 'Create Event'}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
