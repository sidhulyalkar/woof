'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  ChevronLeft,
  Clock3,
  Loader2,
  LogOut,
  MapPin,
  PawPrint,
  Radar,
  Route,
  ShieldCheck,
  Trash2,
  User,
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';
import { BottomNav } from '@/components/bottom-nav';
import { ConnectedServicesSettings } from '@/components/settings/connected-services-settings';
import { NotificationSettings } from '@/components/settings/notification-settings';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { authApi } from '@/lib/api';
import {
  type MeetupLocationSharing,
  privacyApi,
  type PrivacyPreferences,
} from '@/lib/api/privacy';
import type { AuthUser } from '@/lib/stores/auth-store';

const retentionOptions = [1, 6, 12, 24] as const;

function formatStoredDate(value: string | null) {
  if (!value) return 'None';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Please try again.';
}

export default function SettingsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();

  const profileQuery = useQuery<AuthUser>({
    queryKey: ['auth-profile'],
    queryFn: authApi.me,
    staleTime: 30_000,
  });
  const privacyQuery = useQuery<PrivacyPreferences>({
    queryKey: ['privacy-preferences'],
    queryFn: privacyApi.preferences,
    staleTime: 15_000,
  });
  const locationQuery = useQuery({
    queryKey: ['privacy-location-summary'],
    queryFn: privacyApi.locationSummary,
    staleTime: 10_000,
  });

  const updatePrivacy = useMutation({
    mutationFn: privacyApi.updatePreferences,
    onSuccess: (result) => {
      queryClient.setQueryData(['privacy-preferences'], result.preferences);
      void queryClient.invalidateQueries({
        queryKey: ['privacy-location-summary'],
      });
      toast.success('Privacy preferences updated');
    },
    onError: (error) => {
      toast.error(`Could not update privacy settings. ${errorMessage(error)}`);
    },
  });

  const clearLocation = useMutation({
    mutationFn: privacyApi.clearLocationHistory,
    onSuccess: (result) => {
      void queryClient.invalidateQueries({
        queryKey: ['privacy-location-summary'],
      });
      toast.success(
        result.deleted > 0
          ? `Deleted ${result.deleted} stored location point${result.deleted === 1 ? '' : 's'}`
          : 'No stored location history remained',
      );
    },
    onError: (error) => {
      toast.error(`Could not delete location history. ${errorMessage(error)}`);
    },
  });

  const preferences = privacyQuery.data;
  const profile = profileQuery.data;
  const locationSummary = locationQuery.data;
  const isSaving = updatePrivacy.isPending;

  const save = (patch: Partial<PrivacyPreferences>) => {
    if (isSaving) return;
    updatePrivacy.mutate(patch);
  };

  const handleLogout = () => {
    authApi.logout();
    queryClient.clear();
    toast.success('Signed out');
    router.replace('/login');
  };

  if (privacyQuery.isLoading || profileQuery.isLoading) {
    return (
      <main
        id="main-content"
        className="flex min-h-screen items-center justify-center"
        role="status"
      >
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
        <span className="sr-only">Loading settings</span>
      </main>
    );
  }

  if (!preferences) {
    return (
      <main
        id="main-content"
        className="mx-auto flex min-h-screen max-w-xl flex-col items-center justify-center px-6 text-center"
      >
        <ShieldCheck className="h-8 w-8 text-primary" aria-hidden="true" />
        <h1 className="mt-4 text-xl font-semibold">Privacy settings unavailable</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Woof could not load your consent state, so location controls remain unavailable rather than guessing.
        </p>
        <Button className="mt-5" onClick={() => privacyQuery.refetch()}>
          Try again
        </Button>
      </main>
    );
  }

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center gap-3 px-4">
          <Button variant="ghost" size="icon" asChild className="rounded-xl">
            <Link href="/profile" aria-label="Back to profile">
              <ChevronLeft className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
          <div>
            <p className="eyebrow">Consent & control</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Settings</h1>
          </div>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl space-y-6 px-4 py-5">
        <Card className="glass rounded-3xl p-5">
          <div className="flex items-center gap-4">
            <Avatar className="h-14 w-14 border border-border">
              <AvatarImage src={profile?.avatarUrl || '/placeholder.svg'} alt="" />
              <AvatarFallback>
                {profile?.handle ? profile.handle.slice(0, 1).toUpperCase() : <PawPrint className="h-5 w-5" />}
              </AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <p className="font-semibold">{profile ? `@${profile.handle}` : 'Woof member'}</p>
              <p className="truncate text-sm text-muted-foreground">
                {profile?.email ?? 'Authenticated account'}
              </p>
            </div>
            <Button variant="outline" size="sm" asChild>
              <Link href="/profile">
                <User className="mr-2 h-4 w-4" aria-hidden="true" />
                Profile
              </Link>
            </Button>
          </div>
        </Card>

        <section className="space-y-3" aria-labelledby="location-privacy-heading">
          <div>
            <p className="eyebrow">Location privacy</p>
            <h2 id="location-privacy-heading" className="mt-1 text-lg font-bold">
              Precise location is opt-in
            </h2>
            <p className="mt-1 text-sm leading-6 text-muted-foreground">
              Woof does not need precise location for matching. When you choose to enable it for proximity features,
              retained points automatically expire within 24 hours.
            </p>
          </div>

          <Card className="surface-soft divide-y divide-border/60 overflow-hidden rounded-2xl">
            <div className="flex items-start justify-between gap-4 p-4">
              <div className="flex min-w-0 gap-3">
                <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <MapPin className="h-4 w-4" aria-hidden="true" />
                </span>
                <div>
                  <label htmlFor="precise-location" className="font-semibold">
                    Precise location
                  </label>
                  <p className="mt-1 text-sm leading-5 text-muted-foreground">
                    Allows temporary location pings for features you explicitly use. Turning this off also deletes
                    retained pings.
                  </p>
                </div>
              </div>
              <Switch
                id="precise-location"
                checked={preferences.preciseLocation}
                disabled={isSaving}
                onCheckedChange={(checked) => save({ preciseLocation: checked })}
                aria-label="Enable precise location"
              />
            </div>

            <div className="flex items-start justify-between gap-4 p-4">
              <div className="flex min-w-0 gap-3">
                <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                  <Radar className="h-4 w-4" aria-hidden="true" />
                </span>
                <div>
                  <label htmlFor="proximity-suggestions" className="font-semibold">
                    Mutual proximity suggestions
                  </label>
                  <p className="mt-1 text-sm leading-5 text-muted-foreground">
                    Only works when both members opt in. Other members never receive your historical coordinates.
                  </p>
                </div>
              </div>
              <Switch
                id="proximity-suggestions"
                checked={preferences.proximitySuggestions}
                disabled={!preferences.preciseLocation || isSaving}
                onCheckedChange={(checked) => save({ proximitySuggestions: checked })}
                aria-label="Enable mutual proximity suggestions"
              />
            </div>

            <div className="flex items-start justify-between gap-4 p-4">
              <div className="flex min-w-0 gap-3">
                <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/10 text-accent">
                  <Route className="h-4 w-4" aria-hidden="true" />
                </span>
                <div>
                  <label htmlFor="share-routes" className="font-semibold">
                    Share activity routes
                  </label>
                  <p className="mt-1 text-sm leading-5 text-muted-foreground">
                    Off keeps detailed walk and activity geometry private even when you share an activity summary.
                  </p>
                </div>
              </div>
              <Switch
                id="share-routes"
                checked={preferences.shareActivityRoutes}
                disabled={isSaving}
                onCheckedChange={(checked) => save({ shareActivityRoutes: checked })}
                aria-label="Share activity routes"
              />
            </div>
          </Card>
        </section>

        <section className="space-y-3" aria-labelledby="retention-heading">
          <div>
            <p className="eyebrow">Data minimization</p>
            <h2 id="retention-heading" className="mt-1 text-lg font-bold">
              Retention and meetup sharing
            </h2>
          </div>

          <Card className="surface-soft space-y-5 rounded-2xl p-4">
            <div>
              <label htmlFor="retention-hours" className="flex items-center gap-2 font-semibold">
                <Clock3 className="h-4 w-4 text-primary" aria-hidden="true" />
                Location retention
              </label>
              <p className="mt-1 text-sm text-muted-foreground">
                Older location pings are pruned automatically. The server enforces a hard 24-hour maximum.
              </p>
              <select
                id="retention-hours"
                value={preferences.locationRetentionHours}
                disabled={isSaving}
                onChange={(event) => save({ locationRetentionHours: Number(event.target.value) })}
                className="mt-3 h-11 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {retentionOptions.map((hours) => (
                  <option key={hours} value={hours}>
                    {hours === 1 ? '1 hour' : `${hours} hours`}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="meetup-location-sharing" className="font-semibold">
                Meetup location sharing
              </label>
              <p className="mt-1 text-sm text-muted-foreground">
                Proposals contain only a coarse venue area. Choose whether more specific coordination is permitted
                after both people confirm.
              </p>
              <select
                id="meetup-location-sharing"
                value={preferences.meetupLocationSharing}
                disabled={isSaving}
                onChange={(event) =>
                  save({ meetupLocationSharing: event.target.value as MeetupLocationSharing })
                }
                className="mt-3 h-11 w-full rounded-xl border border-border bg-background px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <option value="AFTER_CONFIRMATION">After both people confirm</option>
                <option value="NEVER">Never through Woof location features</option>
              </select>
            </div>
          </Card>
        </section>

        <section className="space-y-3" aria-labelledby="footprint-heading">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="eyebrow">Your footprint</p>
              <h2 id="footprint-heading" className="mt-1 text-lg font-bold">
                Stored location metadata
              </h2>
            </div>
            <Badge variant="outline">max 24h</Badge>
          </div>

          <Card className="surface-soft rounded-2xl p-4">
            {locationQuery.isLoading ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                Inspecting retained metadata
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Points stored</p>
                  <p className="mt-1 text-xl font-bold">{locationSummary?.storedLocationPings ?? 0}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Oldest</p>
                  <p className="mt-1 text-sm font-medium">{formatStoredDate(locationSummary?.oldestStoredAt ?? null)}</p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Newest</p>
                  <p className="mt-1 text-sm font-medium">{formatStoredDate(locationSummary?.newestStoredAt ?? null)}</p>
                </div>
              </div>
            )}

            <div className="mt-4 rounded-xl border border-border/70 bg-background/50 p-3 text-xs leading-5 text-muted-foreground">
              This screen intentionally reports counts and timestamps only. It never retrieves your stored coordinates.
            </div>

            <Button
              variant="outline"
              className="mt-4 w-full gap-2 border-destructive/30 text-destructive hover:bg-destructive/10 hover:text-destructive"
              disabled={clearLocation.isPending || (locationSummary?.storedLocationPings ?? 0) === 0}
              onClick={() => clearLocation.mutate()}
            >
              {clearLocation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Trash2 className="h-4 w-4" aria-hidden="true" />
              )}
              Delete all stored location history
            </Button>
          </Card>
        </section>

        <ConnectedServicesSettings />

        <NotificationSettings />

        <section className="space-y-2" aria-label="Account actions">
          <Button variant="outline" className="w-full justify-start gap-2 bg-transparent" asChild>
            <Link href="/demo">
              <ShieldCheck className="h-4 w-4" aria-hidden="true" />
              View the privacy-safe synthetic demo
            </Link>
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive"
            onClick={handleLogout}
          >
            <LogOut className="h-4 w-4" aria-hidden="true" />
            Sign out
          </Button>
        </section>

        <p className="pb-2 text-center text-xs text-muted-foreground">
          Woof beta · privacy choices are enforced by the API, not just this screen.
        </p>
      </main>

      <BottomNav />
    </div>
  );
}
