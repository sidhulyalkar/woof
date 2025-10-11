'use client';

import { useState } from 'react';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { MapPin, MessageCircle, Calendar, Sparkles, X } from 'lucide-react';

interface NudgeCardProps {
  nudge: {
    id: string;
    type: string;
    payload: {
      targetUserId: string;
      targetUserHandle?: string;
      targetUserAvatar?: string;
      reason: string;
      message: string;
      metadata?: {
        distance?: number;
        venueType?: string;
        messageCount?: number;
        petNames?: {
          yours?: string;
          theirs?: string;
        };
      };
    };
    sentAt: string;
  };
  onAccept: (nudgeId: string) => void;
  onDismiss: (nudgeId: string) => void;
}

export function NudgeCard({ nudge, onAccept, onDismiss }: NudgeCardProps) {
  const [isLoading, setIsLoading] = useState(false);

  const handleAccept = async () => {
    setIsLoading(true);
    await onAccept(nudge.id);
    setIsLoading(false);
  };

  const handleDismiss = async () => {
    setIsLoading(true);
    await onDismiss(nudge.id);
    setIsLoading(false);
  };

  const getIcon = () => {
    switch (nudge.payload.reason) {
      case 'proximity':
        return <MapPin className="h-4 w-4" />;
      case 'chat_activity':
        return <MessageCircle className="h-4 w-4" />;
      case 'mutual_availability':
        return <Calendar className="h-4 w-4" />;
      default:
        return <Sparkles className="h-4 w-4" />;
    }
  };

  const getReasonText = () => {
    switch (nudge.payload.reason) {
      case 'proximity':
        return 'Nearby';
      case 'chat_activity':
        return 'Active Chat';
      case 'mutual_availability':
        return 'Available Now';
      default:
        return 'Suggested';
    }
  };

  const formatDistance = (meters?: number) => {
    if (!meters) return '';
    if (meters < 1000) return `${Math.round(meters)}m away`;
    return `${(meters / 1000).toFixed(1)}km away`;
  };

  return (
    <Card className="relative overflow-hidden">
      {/* Dismiss button */}
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 h-8 w-8 rounded-full z-10"
        onClick={handleDismiss}
        disabled={isLoading}
      >
        <X className="h-4 w-4" />
      </Button>

      <CardContent className="pt-6">
        <div className="flex items-start gap-4">
          {/* Avatar */}
          <Avatar className="h-12 w-12">
            <AvatarImage src={nudge.payload.targetUserAvatar} />
            <AvatarFallback>
              {nudge.payload.targetUserHandle?.[0]?.toUpperCase() || '?'}
            </AvatarFallback>
          </Avatar>

          <div className="flex-1 space-y-2">
            {/* Badge and user handle */}
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="gap-1">
                {getIcon()}
                {getReasonText()}
              </Badge>
              {nudge.payload.metadata?.distance && (
                <span className="text-xs text-muted-foreground">
                  {formatDistance(nudge.payload.metadata.distance)}
                </span>
              )}
            </div>

            {/* Message */}
            <p className="text-sm font-medium">{nudge.payload.message}</p>

            {/* Metadata */}
            {nudge.payload.metadata && (
              <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                {nudge.payload.metadata.petNames && (
                  <span>
                    🐾 {nudge.payload.metadata.petNames.yours} & {nudge.payload.metadata.petNames.theirs}
                  </span>
                )}
                {nudge.payload.metadata.venueType && (
                  <span>
                    📍 {nudge.payload.metadata.venueType}
                  </span>
                )}
                {nudge.payload.metadata.messageCount && (
                  <span>
                    💬 {nudge.payload.metadata.messageCount} messages
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </CardContent>

      <CardFooter className="gap-2 pt-0">
        <Button
          variant="outline"
          className="flex-1"
          onClick={handleDismiss}
          disabled={isLoading}
        >
          Not Now
        </Button>
        <Button
          className="flex-1"
          onClick={handleAccept}
          disabled={isLoading}
        >
          {nudge.type === 'meetup' ? 'Meet Up!' : 'Accept'}
        </Button>
      </CardFooter>
    </Card>
  );
}
