'use client';

import { useEffect, useState } from 'react';
import { NudgeCard } from './nudge-card';
import { useToast } from '@/components/ui/use-toast';
import { useRouter } from 'next/navigation';

interface Nudge {
  id: string;
  type: string;
  payload: any;
  sentAt: string;
}

export function NudgesList() {
  const [nudges, setNudges] = useState<Nudge[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();
  const router = useRouter();

  useEffect(() => {
    fetchNudges();
  }, []);

  const fetchNudges = async () => {
    try {
      const response = await fetch('/api/nudges', {
        credentials: 'include',
      });

      if (!response.ok) throw new Error('Failed to fetch nudges');

      const data = await response.json();
      setNudges(data);
    } catch (error) {
      console.error('Error fetching nudges:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAccept = async (nudgeId: string) => {
    try {
      const response = await fetch(`/api/nudges/${nudgeId}/accept`, {
        method: 'PATCH',
        credentials: 'include',
      });

      if (!response.ok) throw new Error('Failed to accept nudge');

      const nudge = await response.json();

      // Remove from list
      setNudges((prev) => prev.filter((n) => n.id !== nudgeId));

      // Show success toast
      toast({
        title: 'Nudge accepted!',
        description: 'Redirecting you to create a meetup...',
      });

      // Redirect based on nudge type
      if (nudge.type === 'meetup') {
        // Redirect to meetup proposal screen with pre-filled data
        router.push(`/meetup/propose?userId=${nudge.payload.targetUserId}`);
      } else if (nudge.type === 'event') {
        router.push(`/events/${nudge.payload.eventId}`);
      }
    } catch (error) {
      console.error('Error accepting nudge:', error);
      toast({
        title: 'Error',
        description: 'Failed to accept nudge. Please try again.',
        variant: 'destructive',
      });
    }
  };

  const handleDismiss = async (nudgeId: string) => {
    try {
      const response = await fetch(`/api/nudges/${nudgeId}/dismiss`, {
        method: 'PATCH',
        credentials: 'include',
      });

      if (!response.ok) throw new Error('Failed to dismiss nudge');

      // Remove from list
      setNudges((prev) => prev.filter((n) => n.id !== nudgeId));

      toast({
        title: 'Nudge dismissed',
      });
    } catch (error) {
      console.error('Error dismissing nudge:', error);
      toast({
        title: 'Error',
        description: 'Failed to dismiss nudge. Please try again.',
        variant: 'destructive',
      });
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2].map((i) => (
          <div key={i} className="h-32 bg-muted animate-pulse rounded-lg" />
        ))}
      </div>
    );
  }

  if (nudges.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">No active suggestions right now.</p>
        <p className="text-sm text-muted-foreground mt-2">
          We'll notify you when there are nearby pets or chat opportunities!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {nudges.map((nudge) => (
        <NudgeCard
          key={nudge.id}
          nudge={nudge}
          onAccept={handleAccept}
          onDismiss={handleDismiss}
        />
      ))}
    </div>
  );
}
