'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Bell, MapPin, Trophy, X } from 'lucide-react';
import { nudgesApi, type Nudge } from '@/lib/api';
import { toast } from 'sonner';

function readMetadataNumber(metadata: Record<string, unknown>, key: string): number | undefined {
  const value = metadata[key];
  return typeof value === 'number' ? value : undefined;
}

function readPetNames(
  metadata: Record<string, unknown>
): { yours: string; theirs: string } | undefined {
  const value = metadata.petNames;
  if (!value || typeof value !== 'object') return undefined;
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.yours !== 'string' || typeof candidate.theirs !== 'string') return undefined;
  return { yours: candidate.yours, theirs: candidate.theirs };
}

export default function NotificationsPage() {
  const [nudges, setNudges] = useState<Nudge[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    void fetchNudges();
  }, []);

  const fetchNudges = async () => {
    try {
      const response = await nudgesApi.getNudges();
      setNudges(response);
    } catch (error) {
      console.error('Failed to fetch nudges:', error);
      toast.error('Failed to load notifications');
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptNudge = async (nudgeId: string) => {
    try {
      await nudgesApi.acceptNudge(nudgeId);
      setNudges((current) => current.filter((nudge) => nudge.id !== nudgeId));
      toast.success('Great! Check your chat for next steps');
    } catch (error) {
      console.error('Failed to accept nudge:', error);
      toast.error('Failed to accept notification');
    }
  };

  const handleDismissNudge = async (nudgeId: string) => {
    try {
      await nudgesApi.dismissNudge(nudgeId);
      setNudges((current) => current.filter((nudge) => nudge.id !== nudgeId));
    } catch (error) {
      console.error('Failed to dismiss nudge:', error);
      toast.error('Failed to dismiss notification');
    }
  };

  const getNudgeIcon = (type: Nudge['type']) => {
    switch (type) {
      case 'meetup':
        return <MapPin className="h-5 w-5" />;
      case 'achievement':
        return <Trophy className="h-5 w-5" />;
      default:
        return <Bell className="h-5 w-5" />;
    }
  };

  const getNudgeColor = (reason: Nudge['payload']['reason']) => {
    switch (reason) {
      case 'proximity':
        return 'bg-blue-500/10 text-blue-500';
      case 'chat_activity':
        return 'bg-green-500/10 text-green-500';
      case 'goal_achievement':
        return 'bg-yellow-500/10 text-yellow-500';
      default:
        return 'bg-gray-500/10 text-gray-500';
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-4">
        <div className="animate-pulse space-y-4">
          {[...Array(3)].map((_, i) => (
            <Card key={i} className="p-6">
              <div className="mb-4 h-4 w-3/4 rounded bg-muted" />
              <div className="h-3 w-1/2 rounded bg-muted" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-3xl p-4">
      <div className="mb-6">
        <h1 className="mb-2 text-3xl font-bold">Notifications</h1>
        <p className="text-muted-foreground">
          Stay updated with personalized suggestions and alerts
        </p>
      </div>

      {nudges.length === 0 ? (
        <Card className="p-12 text-center">
          <Bell className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />
          <h3 className="mb-2 text-lg font-medium">No new notifications</h3>
          <p className="text-muted-foreground">
            We&apos;ll let you know when there are opportunities for meetups or achievements.
          </p>
        </Card>
      ) : (
        <div className="space-y-4">
          {nudges.map((nudge) => {
            const metadata = nudge.payload.metadata;
            const distance = metadata ? readMetadataNumber(metadata, 'distance') : undefined;
            const messageCount = metadata
              ? readMetadataNumber(metadata, 'messageCount')
              : undefined;
            const petNames = metadata ? readPetNames(metadata) : undefined;

            return (
              <Card key={nudge.id} className="p-6">
                <div className="flex items-start gap-4">
                  <div className={`rounded-full p-3 ${getNudgeColor(nudge.payload.reason)}`}>
                    {getNudgeIcon(nudge.type)}
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="mb-2 flex items-start justify-between gap-4">
                      <div>
                        <h3 className="mb-1 font-medium">
                          {nudge.payload.message || 'New suggestion'}
                        </h3>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <Badge variant="outline" className="text-xs">
                            {nudge.payload.reason.replace('_', ' ')}
                          </Badge>
                          <span>·</span>
                          <span>{new Date(nudge.createdAt).toLocaleDateString()}</span>
                        </div>
                      </div>

                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => void handleDismissNudge(nudge.id)}
                        className="flex-shrink-0"
                        aria-label="Dismiss notification"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    {(distance !== undefined || petNames || messageCount !== undefined) && (
                      <div className="mb-4 text-sm text-muted-foreground">
                        {distance !== undefined && <p>Distance: {distance}m away</p>}
                        {petNames && (
                          <p>
                            {petNames.yours} could meet {petNames.theirs}!
                          </p>
                        )}
                        {messageCount !== undefined && <p>{messageCount} messages exchanged</p>}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <Button size="sm" onClick={() => void handleAcceptNudge(nudge.id)}>
                        {nudge.type === 'meetup' ? "Let's meet!" : 'View'}
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void handleDismissNudge(nudge.id)}
                      >
                        Not now
                      </Button>
                    </div>
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
