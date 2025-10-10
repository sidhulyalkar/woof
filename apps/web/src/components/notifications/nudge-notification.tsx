'use client';

import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Bell, MapPin, MessageCircle, Trophy, X } from 'lucide-react';
import { apiClient } from '@/lib/api/client';
import { toast } from 'sonner';

interface NudgeNotificationProps {
  nudge: {
    id: string;
    type: 'meetup' | 'service' | 'event' | 'achievement';
    payload: {
      targetUserId?: string;
      reason: string;
      message?: string;
      location?: { lat: number; lng: number };
      metadata?: Record<string, any>;
    };
    createdAt: string;
  };
  onAccept?: (nudgeId: string) => void;
  onDismiss?: (nudgeId: string) => void;
  compact?: boolean;
}

export function NudgeNotification({
  nudge,
  onAccept,
  onDismiss,
  compact = false,
}: NudgeNotificationProps) {
  const handleAccept = async () => {
    try {
      await apiClient.patch(`/nudges/${nudge.id}/accept`, {});
      toast.success('Great! Check your messages');
      onAccept?.(nudge.id);
    } catch (error) {
      toast.error('Failed to accept');
    }
  };

  const handleDismiss = async () => {
    try {
      await apiClient.patch(`/nudges/${nudge.id}/dismiss`, {});
      onDismiss?.(nudge.id);
    } catch (error) {
      toast.error('Failed to dismiss');
    }
  };

  const getIcon = () => {
    switch (nudge.type) {
      case 'meetup':
        return nudge.payload.reason === 'proximity' ? (
          <MapPin className="h-4 w-4" />
        ) : (
          <MessageCircle className="h-4 w-4" />
        );
      case 'achievement':
        return <Trophy className="h-4 w-4" />;
      default:
        return <Bell className="h-4 w-4" />;
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 bg-accent/50 rounded-lg">
        <div className="p-2 bg-primary/10 rounded-full text-primary">
          {getIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">
            {nudge.payload.message || 'New suggestion'}
          </p>
        </div>
        <div className="flex gap-1">
          <Button size="sm" variant="default" onClick={handleAccept}>
            View
          </Button>
          <Button size="sm" variant="ghost" onClick={handleDismiss}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className="p-2 bg-primary/10 rounded-full text-primary flex-shrink-0">
          {getIcon()}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-2">
            <div className="flex-1">
              <h4 className="font-medium text-sm mb-1">
                {nudge.payload.message || 'New suggestion'}
              </h4>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" className="text-xs">
                  {nudge.payload.reason.replace('_', ' ')}
                </Badge>
                <span>·</span>
                <span>{new Date(nudge.createdAt).toLocaleTimeString()}</span>
              </div>
            </div>
          </div>

          {nudge.payload.metadata && (
            <div className="text-xs text-muted-foreground mb-3">
              {nudge.payload.metadata.distance && (
                <p>📍 {nudge.payload.metadata.distance}m away</p>
              )}
              {nudge.payload.metadata.petNames && (
                <p>
                  🐕 {nudge.payload.metadata.petNames.yours} + {nudge.payload.metadata.petNames.theirs}
                </p>
              )}
            </div>
          )}

          <div className="flex gap-2">
            <Button size="sm" onClick={handleAccept}>
              {nudge.type === 'meetup' ? "Let's go!" : 'View'}
            </Button>
            <Button size="sm" variant="outline" onClick={handleDismiss}>
              Dismiss
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
