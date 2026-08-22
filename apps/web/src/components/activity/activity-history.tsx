'use client';

import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Clock, ChevronRight, MapPin, Users } from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';
import type { Activity, ActivityType } from '@/lib/types';
import { cn } from '@/lib/utils';

interface ActivityHistoryProps {
  activities: Activity[];
}

const activityTypeLabels: Record<ActivityType, string> = {
  walk: 'Walk',
  play: 'Play',
  playdate: 'Playdate',
  training: 'Training',
  vet: 'Vet visit',
  other: 'Other',
};

const activityTypeColors: Record<ActivityType, string> = {
  walk: 'bg-primary/10 text-primary border-primary/20',
  play: 'bg-secondary/10 text-secondary border-secondary/20',
  playdate: 'bg-secondary/10 text-secondary border-secondary/20',
  training: 'bg-accent/10 text-accent border-accent/20',
  vet: 'bg-muted text-muted-foreground border-border',
  other: 'bg-muted text-muted-foreground border-border',
};

export function ActivityHistory({ activities }: ActivityHistoryProps) {
  if (activities.length === 0) {
    return (
      <div className="py-12 text-center">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-muted">
          <MapPin className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="mb-2 text-lg font-semibold">No activities yet</h3>
        <p className="text-sm text-muted-foreground">Start tracking your pet&apos;s activities!</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h2 className="text-lg font-semibold">Recent Activity</h2>

      <div className="space-y-3">
        {activities.map((activity) => (
          <Card
            key={activity.id}
            className="glass cursor-pointer p-4 transition-colors hover:border-primary/50"
          >
            <div className="flex items-start gap-3">
              <div className="flex-1 space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <Badge className={cn('mb-2 border', activityTypeColors[activity.type])}>
                      {activityTypeLabels[activity.type]}
                    </Badge>
                    <p className="text-sm text-muted-foreground">
                      {formatDistanceToNow(new Date(activity.startTime), { addSuffix: true })}
                    </p>
                  </div>
                  <ChevronRight className="h-5 w-5 shrink-0 text-muted-foreground" />
                </div>

                <div className="space-y-2">
                  {activity.distance !== undefined && (
                    <div className="flex items-center gap-2 text-sm">
                      <MapPin className="h-4 w-4 text-muted-foreground" />
                      <span>{activity.distance.toFixed(1)} miles</span>
                    </div>
                  )}

                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span>
                      {Math.floor(activity.duration / 60)} min
                      {activity.endTime && ` • ${format(new Date(activity.startTime), 'h:mm a')}`}
                    </span>
                  </div>

                  {activity.participants && activity.participants.length > 0 && (
                    <div className="flex items-center gap-2 text-sm">
                      <Users className="h-4 w-4 text-muted-foreground" />
                      <span>{activity.participants.length} participants</span>
                    </div>
                  )}
                </div>

                {activity.notes && (
                  <p className="line-clamp-2 text-sm text-muted-foreground">{activity.notes}</p>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
