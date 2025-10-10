'use client';

import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Bell, MapPin, MessageCircle, Trophy, X } from 'lucide-react';
import { apiClient } from '@/lib/api/client';
import { useAuthStore } from '@/lib/stores/auth-store';
import { toast } from 'sonner';

interface Nudge {
  id: string;
  type: 'meetup' | 'service' | 'event' | 'achievement';
  payload: {
    targetUserId?: string;
    reason: 'proximity' | 'chat_activity' | 'mutual_availability' | 'goal_achievement';
    message?: string;
    location?: { lat: number; lng: number };
    metadata?: Record<string, any>;
  };
  createdAt: string;
  dismissed: boolean;
}

export default function NotificationsPage() {
  const [nudges, setNudges] = useState<Nudge[]>([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuthStore();

  useEffect(() => {
    fetchNudges();
  }, []);

  const fetchNudges = async () => {
    try {
      const response = await apiClient.get('/nudges');
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
      await apiClient.patch(`/nudges/${nudgeId}/accept`, {});
      setNudges(nudges.filter((n) => n.id !== nudgeId));
      toast.success('Great! Check your chat for next steps');
      // TODO: Navigate to relevant screen based on nudge type
    } catch (error) {
      toast.error('Failed to accept notification');
    }
  };

  const handleDismissNudge = async (nudgeId: string) => {
    try {
      await apiClient.patch(`/nudges/${nudgeId}/dismiss`, {});
      setNudges(nudges.filter((n) => n.id !== nudgeId));
    } catch (error) {
      toast.error('Failed to dismiss notification');
    }
  };

  const getNudgeIcon = (type: string) => {
    switch (type) {
      case 'meetup':
        return <MapPin className="h-5 w-5" />;
      case 'service':
        return <Bell className="h-5 w-5" />;
      case 'event':
        return <Bell className="h-5 w-5" />;
      case 'achievement':
        return <Trophy className="h-5 w-5" />;
      default:
        return <Bell className="h-5 w-5" />;
    }
  };

  const getNudgeColor = (reason: string) => {
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
              <div className="h-4 bg-muted rounded w-3/4 mb-4" />
              <div className="h-3 bg-muted rounded w-1/2" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 max-w-3xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Notifications</h1>
        <p className="text-muted-foreground">
          Stay updated with personalized suggestions and alerts
        </p>
      </div>

      {nudges.length === 0 ? (
        <Card className="p-12 text-center">
          <Bell className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-lg font-medium mb-2">No new notifications</h3>
          <p className="text-muted-foreground">
            We'll let you know when there are opportunities for meetups or achievements!
          </p>
        </Card>
      ) : (
        <div className="space-y-4">
          {nudges.map((nudge) => (
            <Card key={nudge.id} className="p-6">
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-full ${getNudgeColor(nudge.payload.reason)}`}>
                  {getNudgeIcon(nudge.type)}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div>
                      <h3 className="font-medium mb-1">
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
                      onClick={() => handleDismissNudge(nudge.id)}
                      className="flex-shrink-0"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>

                  {nudge.payload.metadata && (
                    <div className="text-sm text-muted-foreground mb-4">
                      {nudge.payload.metadata.distance && (
                        <p>
                          Distance: {nudge.payload.metadata.distance}m away
                        </p>
                      )}
                      {nudge.payload.metadata.petNames && (
                        <p>
                          {nudge.payload.metadata.petNames.yours} could meet{' '}
                          {nudge.payload.metadata.petNames.theirs}!
                        </p>
                      )}
                      {nudge.payload.metadata.messageCount && (
                        <p>
                          {nudge.payload.metadata.messageCount} messages exchanged
                        </p>
                      )}
                    </div>
                  )}

                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleAcceptNudge(nudge.id)}
                    >
                      {nudge.type === 'meetup' ? "Let's meet!" : 'View'}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleDismissNudge(nudge.id)}
                    >
                      Not now
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
