'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Avatar } from '@/components/ui/avatar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Calendar, MapPin, Users, Clock, Star, Plus } from 'lucide-react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';
import { EventDetailsDialog } from './EventDetailsDialog';
import { CreateEventDialog } from './CreateEventDialog';

interface Event {
  id: string;
  title: string;
  description: string;
  type: string;
  startTime: string;
  endTime: string;
  locationName: string;
  lat: number;
  lng: number;
  maxAttendees?: number;
  tags: string[];
  organizer: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
  rsvps: Array<{
    userId: string;
    status: 'going' | 'maybe' | 'not_going';
  }>;
  _count?: {
    rsvps: number;
  };
}

export function EventsScreen() {
  const [selectedEvent, setSelectedEvent] = useState<Event | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const queryClient = useQueryClient();

  const { data: events, isLoading } = useQuery<Event[]>({
    queryKey: ['events'],
    queryFn: () => apiClient.get<Event[]>('/events'),
  });

  const rsvpMutation = useMutation({
    mutationFn: ({ eventId, status }: { eventId: string; status: 'going' | 'maybe' | 'not_going' }) =>
      apiClient.post(`/events/${eventId}/rsvp`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['events'] });
      toast.success('RSVP updated!');
    },
    onError: (error: any) => {
      toast.error(error.message || 'Failed to RSVP');
    },
  });

  const upcomingEvents = events?.filter(e => new Date(e.startTime) > new Date()) || [];
  const pastEvents = events?.filter(e => new Date(e.startTime) <= new Date()) || [];

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
    }).format(date);
  };

  const getGoingCount = (event: Event) => {
    return event.rsvps.filter(r => r.status === 'going').length;
  };

  const getUserRSVP = (event: Event, userId: string) => {
    return event.rsvps.find(r => r.userId === userId)?.status;
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Calendar className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
          <p className="text-muted-foreground">Loading events...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div>
            <h1 className="text-2xl font-bold">Community Events</h1>
            <p className="text-sm text-muted-foreground">Group walks, park meetups & more</p>
          </div>
          <Button onClick={() => setShowCreateDialog(true)} className="gap-2">
            <Plus className="w-4 h-4" />
            Create Event
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-4xl mx-auto">
          <Tabs defaultValue="upcoming" className="w-full">
            <TabsList className="w-full mb-4">
              <TabsTrigger value="upcoming" className="flex-1">
                Upcoming ({upcomingEvents.length})
              </TabsTrigger>
              <TabsTrigger value="past" className="flex-1">
                Past ({pastEvents.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upcoming" className="space-y-4">
              {upcomingEvents.length === 0 ? (
                <Card className="p-8 text-center">
                  <Calendar className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="text-lg font-semibold mb-2">No upcoming events</h3>
                  <p className="text-muted-foreground mb-4">Be the first to organize a meetup!</p>
                  <Button onClick={() => setShowCreateDialog(true)}>Create Event</Button>
                </Card>
              ) : (
                upcomingEvents.map(event => (
                  <EventCard
                    key={event.id}
                    event={event}
                    onRSVP={(status) => rsvpMutation.mutate({ eventId: event.id, status })}
                    onClick={() => setSelectedEvent(event)}
                  />
                ))
              )}
            </TabsContent>

            <TabsContent value="past" className="space-y-4">
              {pastEvents.length === 0 ? (
                <Card className="p-8 text-center">
                  <Calendar className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-muted-foreground">No past events yet</p>
                </Card>
              ) : (
                pastEvents.map(event => (
                  <EventCard
                    key={event.id}
                    event={event}
                    isPast
                    onClick={() => setSelectedEvent(event)}
                  />
                ))
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* Dialogs */}
      {selectedEvent && (
        <EventDetailsDialog
          event={selectedEvent}
          open={!!selectedEvent}
          onClose={() => setSelectedEvent(null)}
        />
      )}
      <CreateEventDialog
        open={showCreateDialog}
        onClose={() => setShowCreateDialog(false)}
      />
    </div>
  );
}

interface EventCardProps {
  event: Event;
  onRSVP?: (status: 'going' | 'maybe' | 'not_going') => void;
  isPast?: boolean;
  onClick?: () => void;
}

function EventCard({ event, onRSVP, isPast, onClick }: EventCardProps) {
  const goingCount = event.rsvps.filter(r => r.status === 'going').length;
  const spotsLeft = event.maxAttendees ? event.maxAttendees - goingCount : null;

  return (
    <Card
      className="p-6 hover:shadow-lg transition-all cursor-pointer"
      onClick={onClick}
    >
      <div className="flex gap-4">
        {/* Date Badge */}
        <div className="flex-shrink-0">
          <div className="w-16 h-16 rounded-lg bg-accent/10 flex flex-col items-center justify-center">
            <div className="text-xs text-accent font-semibold">
              {new Date(event.startTime).toLocaleDateString('en-US', { month: 'short' }).toUpperCase()}
            </div>
            <div className="text-2xl font-bold text-accent">
              {new Date(event.startTime).getDate()}
            </div>
          </div>
        </div>

        <div className="flex-1 min-w-0">
          {/* Title & Type */}
          <div className="flex items-start justify-between gap-2 mb-2">
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-bold truncate">{event.title}</h3>
              <Badge variant="secondary" className="text-xs">
                {event.type}
              </Badge>
            </div>
            {!isPast && spotsLeft !== null && spotsLeft <= 5 && (
              <Badge variant="destructive" className="text-xs flex-shrink-0">
                {spotsLeft} spots left
              </Badge>
            )}
          </div>

          {/* Description */}
          <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
            {event.description}
          </p>

          {/* Details */}
          <div className="flex flex-wrap gap-3 text-sm text-muted-foreground mb-3">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              {new Date(event.startTime).toLocaleTimeString('en-US', {
                hour: 'numeric',
                minute: '2-digit',
              })}
            </div>
            <div className="flex items-center gap-1">
              <MapPin className="w-4 h-4" />
              {event.locationName}
            </div>
            <div className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              {goingCount} going
              {event.maxAttendees && ` / ${event.maxAttendees}`}
            </div>
          </div>

          {/* Organizer & Tags */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Avatar className="w-6 h-6">
                {event.organizer.avatarUrl ? (
                  <img src={event.organizer.avatarUrl} alt={event.organizer.handle} />
                ) : (
                  <div className="bg-accent text-white flex items-center justify-center text-xs">
                    {event.organizer.handle[0].toUpperCase()}
                  </div>
                )}
              </Avatar>
              <span className="text-xs text-muted-foreground">@{event.organizer.handle}</span>
            </div>

            {!isPast && onRSVP && (
              <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-8"
                  onClick={() => onRSVP('maybe')}
                >
                  Maybe
                </Button>
                <Button
                  size="sm"
                  className="h-8 gap-1"
                  onClick={() => onRSVP('going')}
                >
                  <Users className="w-3 h-3" />
                  Going
                </Button>
              </div>
            )}

            {isPast && (
              <Button
                size="sm"
                variant="outline"
                className="h-8 gap-1"
                onClick={(e) => {
                  e.stopPropagation();
                  onClick?.();
                }}
              >
                <Star className="w-3 h-3" />
                Leave Feedback
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
