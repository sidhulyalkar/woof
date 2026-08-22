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
      metadata?: {
        distance?: number;
        petNames?: { yours?: string; theirs?: string };
        [key: string]: unknown;
      };
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
      await apiClient.patch<void>(`/nudges/${nudge.id}/accept`, {});
      toast.success('Great! Check your messages');
      onAccept?.(nudge.id);
    } catch {
      toast.error('Failed to accept');
    }
  };

  const handleDismiss = async () => {
    try {
      await apiClient.patch<void>(`/nudges/${nudge.id}/dismiss`, {});
      onDismiss?.(nudge.id);
    } catch {
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
      <div className="flex items-center gap-3 rounded-lg bg-accent/50 p-3">
        <div className="rounded-full bg-primary/10 p-2 text-primary">{getIcon()}</div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">
            {nudge.payload.message || 'New suggestion'}
          </p>
        </div>
        <div className="flex gap-1">
          <Button size="sm" variant="default" onClick={() => void handleAccept()}>
            View
          </Button>
          <Button size="sm" variant="ghost" onClick={() => void handleDismiss()}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Card className="p-4">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 rounded-full bg-primary/10 p-2 text-primary">
          {getIcon()}
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-start justify-between">
            <div className="flex-1">
              <h4 className="mb-1 text-sm font-medium">
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
            <div className="mb-3 text-xs text-muted-foreground">
              {nudge.payload.metadata.distance !== undefined && (
                <p>📍 {nudge.payload.metadata.distance}m away</p>
              )}
              {nudge.payload.metadata.petNames && (
                <p>
                  🐕 {nudge.payload.metadata.petNames.yours} +{' '}
                  {nudge.payload.metadata.petNames.theirs}
                </p>
              )}
            </div>
          )}

          <div className="flex gap-2">
            <Button size="sm" onClick={() => void handleAccept()}>
              {nudge.type === 'meetup' ? "Let's go!" : 'View'}
            </Button>
            <Button size="sm" variant="outline" onClick={() => void handleDismiss()}>
              Dismiss
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
}
