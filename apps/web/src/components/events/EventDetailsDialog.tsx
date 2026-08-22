'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarImage } from '@/components/ui/avatar';
import { Slider } from '@/components/ui/slider';
import { MapPin, Clock, Users, Star, Send } from 'lucide-react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

type LegacyEventDetails = {
  id: string;
  title: string;
  type: string;
  description: string;
  startTime: string;
  endTime: string;
  locationName: string;
  maxAttendees?: number;
  rsvps: Array<{ status: string }>;
  organizer: { handle: string; avatarUrl?: string | null };
  tags?: string[];
};

type EventFeedbackResponse = {
  feedback: unknown[];
  averages: {
    vibeScore: number;
    petDensity: number;
    venueQuality: number;
  };
  totalFeedback: number;
};

interface EventDetailsDialogProps {
  event: LegacyEventDetails;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EventDetailsDialog({ event, open, onOpenChange }: EventDetailsDialogProps) {
  const queryClient = useQueryClient();
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState({
    vibeScore: 3,
    petDensity: 3,
    venueQuality: 3,
    notes: '',
  });

  const { data: feedbackData } = useQuery<EventFeedbackResponse>({
    queryKey: ['event-feedback', event.id],
    queryFn: () => apiClient.get<EventFeedbackResponse>(`/events/${event.id}/feedback`),
    enabled: open,
  });

  const rsvpMutation = useMutation({
    mutationFn: (status: 'going' | 'maybe' | 'not_going') =>
      apiClient.post(`/events/${event.id}/rsvp`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events'] });
      toast.success('RSVP updated!');
    },
    onError: () => toast.error('Failed to update RSVP'),
  });

  const checkInMutation = useMutation({
    mutationFn: () => apiClient.post(`/events/${event.id}/checkin`),
    onSuccess: () => {
      toast.success('Checked in! Enjoy the event 🎉');
      queryClient.invalidateQueries({ queryKey: ['events'] });
    },
    onError: () => toast.error('Failed to check in'),
  });

  const feedbackMutation = useMutation({
    mutationFn: () =>
      apiClient.post<void>(`/events/${event.id}/feedback`, {
        vibeScore: feedback.vibeScore,
        petDensity: feedback.petDensity,
        venueQuality: feedback.venueQuality,
        notes: feedback.notes || undefined,
      }),
    onSuccess: () => {
      toast.success('Thanks for the feedback!');
      setShowFeedback(false);
      queryClient.invalidateQueries({ queryKey: ['event-feedback', event.id] });
    },
    onError: () => toast.error('Failed to submit feedback'),
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div>
              <Badge className="mb-2" variant="secondary">
                {event.type}
              </Badge>
              <DialogTitle className="text-2xl">{event.title}</DialogTitle>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          <p className="text-muted-foreground">{event.description}</p>

          <div className="rounded-lg bg-muted/50 p-4">
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span>
                  {new Date(event.startTime).toLocaleTimeString('en-US', {
                    hour: 'numeric',
                    minute: '2-digit',
                  })}{' '}
                  -{' '}
                  {new Date(event.endTime).toLocaleTimeString('en-US', {
                    hour: 'numeric',
                    minute: '2-digit',
                  })}
                </span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <MapPin className="h-4 w-4 text-muted-foreground" />
                <span>{event.locationName}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Users className="h-4 w-4 text-muted-foreground" />
                <span>
                  {event.rsvps.filter((rsvp) => rsvp.status === 'going').length} going
                  {event.maxAttendees && ` / ${event.maxAttendees} max`}
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="mb-2 text-sm font-semibold">Organized by</h4>
            <div className="flex items-center gap-2">
              <Avatar className="h-8 w-8">
                {event.organizer.avatarUrl ? (
                  <AvatarImage src={event.organizer.avatarUrl} alt={event.organizer.handle} />
                ) : (
                  <div className="flex items-center justify-center bg-accent text-sm text-white">
                    {event.organizer.handle[0].toUpperCase()}
                  </div>
                )}
              </Avatar>
              <span className="text-sm font-medium">@{event.organizer.handle}</span>
            </div>
          </div>

          {event.tags && event.tags.length > 0 && (
            <div>
              <h4 className="mb-2 text-sm font-semibold">Tags</h4>
              <div className="flex flex-wrap gap-2">
                {event.tags.map((tag) => (
                  <Badge key={tag} variant="outline">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {feedbackData && feedbackData.totalFeedback > 0 && (
            <div className="rounded-lg border p-4">
              <h4 className="mb-3 font-semibold">Community feedback</h4>
              <div className="grid grid-cols-3 gap-4 text-center text-sm">
                <div>
                  <div className="font-semibold">{feedbackData.averages.vibeScore.toFixed(1)}</div>
                  <div className="text-muted-foreground">Vibe</div>
                </div>
                <div>
                  <div className="font-semibold">{feedbackData.averages.petDensity.toFixed(1)}</div>
                  <div className="text-muted-foreground">Pet density</div>
                </div>
                <div>
                  <div className="font-semibold">
                    {feedbackData.averages.venueQuality.toFixed(1)}
                  </div>
                  <div className="text-muted-foreground">Venue</div>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-3 gap-2">
            <Button
              variant="outline"
              onClick={() => rsvpMutation.mutate('maybe')}
              disabled={rsvpMutation.isPending}
            >
              Maybe
            </Button>
            <Button
              onClick={() => rsvpMutation.mutate('going')}
              disabled={rsvpMutation.isPending}
            >
              Going
            </Button>
            <Button
              variant="outline"
              onClick={() => rsvpMutation.mutate('not_going')}
              disabled={rsvpMutation.isPending}
            >
              Can&apos;t go
            </Button>
          </div>

          <Button
            className="w-full"
            variant="secondary"
            onClick={() => checkInMutation.mutate()}
            disabled={checkInMutation.isPending}
          >
            Check in
          </Button>

          {!showFeedback ? (
            <Button variant="ghost" className="w-full" onClick={() => setShowFeedback(true)}>
              Share event feedback
            </Button>
          ) : (
            <div className="space-y-5 rounded-lg border p-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Vibe</span>
                  <span>{feedback.vibeScore}/5</span>
                </div>
                <Slider
                  value={[feedback.vibeScore]}
                  min={1}
                  max={5}
                  step={1}
                  onValueChange={([vibeScore]) => setFeedback((value) => ({ ...value, vibeScore }))}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Pet density</span>
                  <span>{feedback.petDensity}/5</span>
                </div>
                <Slider
                  value={[feedback.petDensity]}
                  min={1}
                  max={5}
                  step={1}
                  onValueChange={([petDensity]) => setFeedback((value) => ({ ...value, petDensity }))}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Venue quality</span>
                  <span>{feedback.venueQuality}/5</span>
                </div>
                <Slider
                  value={[feedback.venueQuality]}
                  min={1}
                  max={5}
                  step={1}
                  onValueChange={([venueQuality]) =>
                    setFeedback((value) => ({ ...value, venueQuality }))
                  }
                />
              </div>
              <textarea
                className="min-h-24 w-full rounded-md border bg-background p-3 text-sm"
                value={feedback.notes}
                onChange={(event) =>
                  setFeedback((value) => ({ ...value, notes: event.target.value }))
                }
                placeholder="Anything useful for future attendees?"
              />
              <Button
                className="w-full"
                onClick={() => feedbackMutation.mutate()}
                disabled={feedbackMutation.isPending}
              >
                <Send className="mr-2 h-4 w-4" />
                Submit feedback
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
