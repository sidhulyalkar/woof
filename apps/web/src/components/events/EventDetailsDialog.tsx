'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar } from '@/components/ui/avatar';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { MapPin, Clock, Users, Star, Send } from 'lucide-react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

interface EventDetailsDialogProps {
  event: any;
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

  const { data: feedback } = useQuery({
    queryKey: ['event-feedback', event.id],
    queryFn: () => apiClient.get(`/events/${event.id}/feedback`),
    enabled: isPast,
  });

  const submitFeedbackMutation = useMutation({
    mutationFn: (data: any) => apiClient.post(`/events/${event.id}/feedback`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['event-feedback', event.id] });
      toast.success('Feedback submitted! Thanks for helping improve future events.');
      setShowFeedbackForm(false);
    },
    onError: (error: any) => {
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
    setFeedbackTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
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
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{event.title}</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {/* Event Details */}
          <div>
            <Badge variant="secondary" className="mb-3">{event.type}</Badge>
            <p className="text-muted-foreground mb-4">{event.description}</p>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-muted-foreground" />
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
                <Clock className="w-4 h-4 text-muted-foreground" />
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
                <MapPin className="w-4 h-4 text-muted-foreground" />
                <span>{event.locationName}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Users className="w-4 h-4 text-muted-foreground" />
                <span>
                  {event.rsvps.filter((r: any) => r.status === 'going').length} going
                  {event.maxAttendees && ` / ${event.maxAttendees} max`}
                </span>
              </div>
            </div>
          </div>

          {/* Organizer */}
          <div>
            <h4 className="text-sm font-semibold mb-2">Organized by</h4>
            <div className="flex items-center gap-2">
              <Avatar className="w-8 h-8">
                {event.organizer.avatarUrl ? (
                  <img src={event.organizer.avatarUrl} alt={event.organizer.handle} />
                ) : (
                  <div className="bg-accent text-white flex items-center justify-center text-sm">
                    {event.organizer.handle[0].toUpperCase()}
                  </div>
                )}
              </Avatar>
              <span className="text-sm font-medium">@{event.organizer.handle}</span>
            </div>
          </div>

          {/* Tags */}
          {event.tags && event.tags.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Tags</h4>
              <div className="flex flex-wrap gap-2">
                {event.tags.map((tag: string, idx: number) => (
                  <Badge key={idx} variant="outline">{tag}</Badge>
                ))}
              </div>
            </div>
          )}

          {/* Past Event: Feedback */}
          {isPast && (
            <div className="border-t pt-6">
              {!showFeedbackForm ? (
                <div>
                  <h4 className="text-sm font-semibold mb-3">Event Feedback</h4>
                  {feedback && feedback.feedback.length > 0 ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg">
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
                        {feedback.totalFeedback} {feedback.totalFeedback === 1 ? 'person' : 'people'} rated this event
                      </p>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground mb-4">No feedback yet</p>
                  )}
                  <Button onClick={() => setShowFeedbackForm(true)} className="mt-4 gap-2">
                    <Star className="w-4 h-4" />
                    Leave Feedback
                  </Button>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <h4 className="text-sm font-semibold mb-4">Rate this event</h4>

                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-2">
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
                        <div className="flex justify-between mb-2">
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
                        <div className="flex justify-between mb-2">
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
                    <h4 className="text-sm font-semibold mb-2">Quick tags (optional)</h4>
                    <div className="flex flex-wrap gap-2">
                      {suggestedTags.map(tag => (
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
                      <Send className="w-4 h-4" />
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
