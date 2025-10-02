'use client';

import { useQuery } from '@tanstack/react-query';
import { socialApi } from '@/lib/api';
import { SocialFeed } from '@/components/social/social-feed';
import { HappinessMetrics } from '@/components/social/happiness-metrics';

export default function SocialPage() {
  const { data: feed, isLoading } = useQuery({
    queryKey: ['social-feed'],
    queryFn: socialApi.getFeed,
  });

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="space-y-6">
        <HappinessMetrics />
        <SocialFeed posts={feed || []} isLoading={isLoading} />
      </div>
    </div>
  );
}
