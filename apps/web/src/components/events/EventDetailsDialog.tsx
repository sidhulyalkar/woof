'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar } from '@/components/ui/avatar';
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

type FeedbackSubmission = {
  vibeScore: number;
  petDensity: number;
  venueQuality: number;
  tags: string[];
};

interface EventDetailsDialogProps {
  event: LegacyEventDetails;
  open: boolean;
  onClose: () => void;
}

export function EventDetailsDialog({ event, open, onClose }: EventDetailsDialogProps) {
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [vibeScore, setVibeScore] = useState([5]);
  const [petDensity, setPetDensity] = useState([5]);
  const [venueQuality, setVenueQuality] = useState([5]);
  const [feedbackTags, setFeedbackTags] = useState<string[]>([]);
  const queryClient = useQueryClient();

  const isPast = new Date(event.startTime) < new Date();

  const { data: feedback } = useQuery<EventFeedbackResponse>({
    queryKey: ['event-feedback', event.id],
    queryFn: () => apiClient.get<EventFeedbackResponse>(`/events/${event.id}/feedback`),
    enabled: isPast,
  });

  const submitFeedbackMutation = useMutation<void, Error, FeedbackSubmission>({
    mutationFn: (data) => apiClient.post<void>(`/events/${event.id}/feedback`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['event-feedback', event.id] });
      toast.success('Feedback submitted! Thanks for helping improve future events.');
      setShowFeedbackForm(false);
    },
    onError: (error) => {
      toast.error(error.message || 'Failed to submit feedback');
    },
  });

  const handleSubmitFeedback = () => {
    submitFeedbackMutation.mutate({
      vibeScore: vibeScore[0],
      petDensity: petDensity[0],
      venueQuality: venueQuality[0],
      tags: feedbackTags,
    });
  };

  const toggleTag = (tag: string) => {
    setFeedbackTags((previous) =>
      previous.includes(tag) ? previous.filter((item) => item !== tag) : [...previous, tag]
    );
  };

  const suggestedTags = [
    'Well organized',
    'Great location',
    'Friendly crowd',
    'Good turnout',
    'Would attend again',
    'Too crowded',
    'Hard to find',
  ];

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <DialogContent className="max-h-[90vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{event.title}</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          <div>
            <Badge variant="secondary" className="mb-3">
              {event.type}
            </Badge>
            <p className="mb-4 text-muted-foreground">{event.description}</p>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span>
                  {new Date(event.startTime).toLocaleDateString('en-US', {
                    weekday: 'long',
                    month: 'long',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                </span>
              </div>
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
                  <img src={event.organizer.avatarUrl} alt={event.organizer.handle} />
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

          {isPast && (
            <div className="border-t pt-6">
              {!showFeedbackForm ? (
                <div>
                  <h4 className="mb-3 text-sm font-semibold">Event Feedback</h4>
                  {feedback && feedback.feedback.length > 0 ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4 rounded-lg bg-muted/50 p-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-accent">
                            {feedback.averages.vibeScore.toFixed(1)}
                          </div>
                          <div className="text-xs text-muted-foreground">Vibe</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-accent">
                            {feedback.averages.petDensity.toFixed(1)}
                          </div>
                          <div className="text-xs text-muted-foreground">Pet Density</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-accent">
                            {feedback.averages.venueQuality.toFixed(1)}
                          </div>
                          <div className="text-xs text-muted-foreground">Venue</div>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {feedback.totalFeedback}{' '}
                        {feedback.totalFeedback === 1 ? 'person' : 'people'} rated this event
                      </p>
                    </div>
                  ) : (
                    <p className="mb-4 text-sm text-muted-foreground">No feedback yet</p>
                  )}
                  <Button onClick={() => setShowFeedbackForm(true)} className="mt-4 gap-2">
                    <Star className="h-4 w-4" />
                    Leave Feedback
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h4 className="mb-4 text-sm font-semibold">Rate this event</h4>

                    <div className="space-y-4">
                      <div>
                        <div className="mb-2 flex justify-between">
                          <label className="text-sm">Vibe Score</label>
                          <span className="text-sm font-semibold">{vibeScore[0]}/5</span>
                        </div>
                        <Slider
                          value={vibeScore}
                          onValueChange={setVibeScore}
                          min={1}
                          max={5}
                          step={1}
                        />
                      </div>

                      <div>
                        <div className="mb-2 flex justify-between">
                          <label className="text-sm">Pet Density</label>
                          <span className="text-sm font-semibold">{petDensity[0]}/5</span>
                        </div>
                        <Slider
                          value={petDensity}
                          onValueChange={setPetDensity}
                          min={1}
                          max={5}
                          step={1}
                        />
                      </div>

                      <div>
                        <div className="mb-2 flex justify-between">
                          <label className="text-sm">Venue Quality</label>
                          <span className="text-sm font-semibold">{venueQuality[0]}/5</span>
                        </div>
                        <Slider
                          value={venueQuality}
                          onValueChange={setVenueQuality}
                          min={1}
                          max={5}
                          step={1}
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="mb-2 text-sm font-semibold">Quick tags (optional)</h4>
                    <div className="flex flex-wrap gap-2">
                      {suggestedTags.map((tag) => (
                        <Badge
                          key={tag}
                          variant={feedbackTags.includes(tag) ? 'default' : 'outline'}
                          className="cursor-pointer"
                          onClick={() => toggleTag(tag)}
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => setShowFeedbackForm(false)}
                      className="flex-1"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={handleSubmitFeedback}
                      disabled={submitFeedbackMutation.isPending}
                      className="flex-1 gap-2"
                    >
                      <Send className="h-4 w-4" />
                      Submit
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
