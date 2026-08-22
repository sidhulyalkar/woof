'use client';

import Image from 'next/image';
import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Avatar, AvatarImage } from '@/components/ui/avatar';
import { Heart, X, Star, MapPin, Clock, Sparkles } from 'lucide-react';
import { useGetMatches, useRecordInteraction } from '@/lib/api/hooks';
import { toast } from 'sonner';

export function MatchDiscoveryScreen() {
  const { data: matches, isLoading, refetch } = useGetMatches();
  const recordInteraction = useRecordInteraction();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [swipeDirection, setSwipeDirection] = useState<'left' | 'right' | null>(null);

  const currentMatch = matches?.[currentIndex];

  const handleSwipe = async (action: 'like' | 'skip' | 'super_like') => {
    if (!currentMatch) return;

    setSwipeDirection(action === 'skip' ? 'left' : 'right');

    try {
      await recordInteraction.mutateAsync({
        targetUserId: currentMatch.user.id,
        action,
      });

      if (action === 'like' || action === 'super_like') {
        toast.success(`You liked ${currentMatch.pet.name}!`);
      }

      setTimeout(() => {
        setSwipeDirection(null);
        setCurrentIndex((previous) => previous + 1);

        if (matches && currentIndex >= matches.length - 2) {
          void refetch();
        }
      }, 300);
    } catch (error) {
      console.error('Failed to record interaction:', error);
      toast.error('Failed to process swipe');
      setSwipeDirection(null);
    }
  };

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Sparkles className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
          <p className="text-muted-foreground">Finding your perfect matches...</p>
        </div>
      </div>
    );
  }

  if (!matches || matches.length === 0 || !currentMatch) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <Card className="p-8 text-center max-w-md">
          <Sparkles className="w-16 h-16 mx-auto mb-4 text-accent" />
          <h3 className="text-xl font-bold mb-2">No more matches right now</h3>
          <p className="text-muted-foreground mb-4">
            Check back soon! We&apos;re finding more paw-fect matches for you.
          </p>
          <Button
            onClick={() => {
              setCurrentIndex(0);
              void refetch();
            }}
          >
            Refresh Matches
          </Button>
        </Card>
      </div>
    );
  }

  const match = currentMatch;

  return (
    <div className="h-full flex flex-col bg-gradient-to-b from-background to-muted/20">
      <div className="p-4 border-b border-border/20">
        <div className="flex items-center justify-between max-w-2xl mx-auto">
          <div>
            <h1 className="text-2xl font-bold">Discover</h1>
            <p className="text-sm text-muted-foreground">Find your next walking buddy</p>
          </div>
          <Badge variant="secondary" className="text-lg px-4 py-1">
            <Sparkles className="w-4 h-4 mr-1" />
            {matches.length - currentIndex} left
          </Badge>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center p-4">
        <div
          className={`w-full max-w-2xl transition-all duration-300 ${
            swipeDirection === 'left'
              ? 'transform -translate-x-full opacity-0'
              : swipeDirection === 'right'
                ? 'transform translate-x-full opacity-0'
                : 'transform translate-x-0 opacity-100'
          }`}
        >
          <Card className="overflow-hidden shadow-2xl">
            <div className="relative h-96 bg-gradient-to-br from-accent/20 to-primary/20">
              {match.pet.avatarUrl ? (
                <Image
                  src={match.pet.avatarUrl}
                  alt={match.pet.name}
                  fill
                  sizes="(max-width: 768px) 100vw, 672px"
                  className="object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-8xl">🐕</div>
              )}

              <div className="absolute top-4 right-4">
                <Badge className="bg-white/90 text-accent border-2 border-accent text-lg px-4 py-2">
                  <Heart className="w-5 h-5 mr-1 fill-current" />
                  {match.compatibilityScore}% Match
                </Badge>
              </div>

              {match.distance && (
                <div className="absolute top-4 left-4">
                  <Badge variant="secondary" className="bg-white/90 text-sm">
                    <MapPin className="w-3 h-3 mr-1" />
                    {match.distance < 1
                      ? `${(match.distance * 1000).toFixed(0)}m`
                      : `${match.distance.toFixed(1)}km`}{' '}
                    away
                  </Badge>
                </div>
              )}
            </div>

            <div className="p-6 space-y-4">
              <div>
                <h2 className="text-3xl font-bold">
                  {match.pet.name}, {match.pet.age}
                </h2>
                <p className="text-lg text-muted-foreground">{match.pet.breed}</p>
                <div className="flex items-center gap-2 mt-2">
                  <Avatar className="w-8 h-8">
                    {match.user.avatarUrl ? (
                      <AvatarImage src={match.user.avatarUrl} alt={match.user.handle} />
                    ) : (
                      <div className="bg-accent text-white flex items-center justify-center text-sm">
                        {match.user.handle[0].toUpperCase()}
                      </div>
                    )}
                  </Avatar>
                  <span className="text-sm text-muted-foreground">@{match.user.handle}</span>
                  {match.lastActive && (
                    <>
                      <span className="text-muted-foreground">•</span>
                      <span className="text-sm text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Active {match.lastActive}
                      </span>
                    </>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2 flex items-center gap-1">
                  <Sparkles className="w-4 h-4 text-accent" />
                  Why you&apos;ll get along
                </p>
                <div className="flex flex-wrap gap-2">
                  {match.explainability.topReasons.map((reason, index) => (
                    <Badge key={index} variant="outline" className="text-sm">
                      {reason}
                    </Badge>
                  ))}
                  {match.explainability.mutualInterests?.map((interest, index) => (
                    <Badge key={`interest-${index}`} variant="secondary" className="text-sm">
                      {interest}
                    </Badge>
                  ))}
                </div>
              </div>

              {match.user.bio && (
                <div>
                  <p className="text-sm text-muted-foreground">{match.user.bio}</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>

      <div className="p-6 border-t border-border/20">
        <div className="flex items-center justify-center gap-4 max-w-md mx-auto">
          <Button
            size="lg"
            variant="outline"
            className="w-16 h-16 rounded-full border-2 hover:border-destructive hover:bg-destructive/10"
            onClick={() => handleSwipe('skip')}
            disabled={recordInteraction.isPending}
          >
            <X className="w-6 h-6 text-destructive" />
          </Button>

          <Button
            size="lg"
            variant="outline"
            className="w-20 h-20 rounded-full border-2 border-accent hover:bg-accent/10 hover:scale-110 transition-transform"
            onClick={() => handleSwipe('super_like')}
            disabled={recordInteraction.isPending}
          >
            <Star className="w-8 h-8 text-accent fill-current" />
          </Button>

          <Button
            size="lg"
            className="w-16 h-16 rounded-full bg-accent hover:bg-accent/90 hover:scale-110 transition-transform"
            onClick={() => handleSwipe('like')}
            disabled={recordInteraction.isPending}
          >
            <Heart className="w-6 h-6 fill-current" />
          </Button>
        </div>
      </div>
    </div>
  );
}
