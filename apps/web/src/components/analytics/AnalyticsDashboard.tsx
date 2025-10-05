'use client';

import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Users,
  Calendar,
  Target,
  Zap,
  Heart,
  ShoppingBag,
  Award,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

interface NorthStarMetrics {
  meetupConversionRate: number;
  meetupConversionRateTrend: number;
  retention7Day: number;
  retention7DayTrend: number;
  dataYieldPerUser: number;
  dataYieldPerUserTrend: number;
  serviceIntentRate: number;
  serviceIntentRateTrend: number;
  totalUsers: number;
  activeUsers: number;
  totalMeetups: number;
  completedMeetups: number;
}

interface MetricDetail {
  totalMatches: number;
  uniqueMatchChats: number;
  meetupsProposed: number;
  meetupsConfirmed: number;
  meetupsCompleted: number;
  avgFeedbackPerUser: number;
  serviceIntents: {
    total: number;
    tapBook: number;
    conversions: number;
  };
  eventsFeedback: {
    total: number;
    avgVibeScore: number;
    avgPetDensity: number;
    avgVenueQuality: number;
  };
}

export function AnalyticsDashboard() {
  const [timeframe, setTimeframe] = useState<'7d' | '30d' | '90d'>('30d');

  const { data: metrics, isLoading } = useQuery<NorthStarMetrics>({
    queryKey: ['analytics', 'north-star', timeframe],
    queryFn: () => apiClient.get<NorthStarMetrics>(`/analytics/north-star?timeframe=${timeframe}`),
  });

  const { data: details } = useQuery<MetricDetail>({
    queryKey: ['analytics', 'details', timeframe],
    queryFn: () => apiClient.get<MetricDetail>(`/analytics/details?timeframe=${timeframe}`),
  });

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <BarChart3 className="w-12 h-12 mx-auto mb-4 text-accent animate-pulse" />
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border/20">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
              <p className="text-sm text-muted-foreground">Track north star metrics & product health</p>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant={timeframe === '7d' ? 'default' : 'outline'}
                onClick={() => setTimeframe('7d')}
              >
                7 Days
              </Button>
              <Button
                size="sm"
                variant={timeframe === '30d' ? 'default' : 'outline'}
                onClick={() => setTimeframe('30d')}
              >
                30 Days
              </Button>
              <Button
                size="sm"
                variant={timeframe === '90d' ? 'default' : 'outline'}
                onClick={() => setTimeframe('90d')}
              >
                90 Days
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-4">
        <div className="max-w-6xl mx-auto space-y-6">
          {/* North Star Metrics */}
          <div>
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-accent" />
              North Star Metrics
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Meetup Conversion"
                value={`${metrics?.meetupConversionRate.toFixed(1)}%`}
                trend={metrics?.meetupConversionRateTrend}
                description="Meetups confirmed / unique match chats"
                icon={Calendar}
                color="blue"
              />
              <MetricCard
                title="7D Retention"
                value={`${metrics?.retention7Day.toFixed(1)}%`}
                trend={metrics?.retention7DayTrend}
                description="Users returning within 7 days"
                icon={Users}
                color="green"
              />
              <MetricCard
                title="Data Yield/User"
                value={metrics?.dataYieldPerUser.toFixed(1) || '0'}
                trend={metrics?.dataYieldPerUserTrend}
                description="Labeled interactions + outcomes"
                icon={Zap}
                color="purple"
              />
              <MetricCard
                title="Service Intent Rate"
                value={`${metrics?.serviceIntentRate.toFixed(1)}%`}
                trend={metrics?.serviceIntentRateTrend}
                description="Service taps to book / MAU"
                icon={ShoppingBag}
                color="orange"
              />
            </div>
          </div>

          {/* Key Stats */}
          <div>
            <h2 className="text-lg font-bold mb-4">Key Statistics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Total Users"
                value={metrics?.totalUsers || 0}
                icon={Users}
              />
              <StatCard
                label="Active Users"
                value={metrics?.activeUsers || 0}
                icon={Users}
                highlight
              />
              <StatCard
                label="Total Meetups"
                value={metrics?.totalMeetups || 0}
                icon={Calendar}
              />
              <StatCard
                label="Completed Meetups"
                value={metrics?.completedMeetups || 0}
                icon={Calendar}
                highlight
              />
            </div>
          </div>

          {/* Detailed Metrics */}
          <Tabs defaultValue="meetups" className="w-full">
            <TabsList className="w-full mb-4">
              <TabsTrigger value="meetups" className="flex-1">Meetups</TabsTrigger>
              <TabsTrigger value="services" className="flex-1">Services</TabsTrigger>
              <TabsTrigger value="events" className="flex-1">Events</TabsTrigger>
              <TabsTrigger value="engagement" className="flex-1">Engagement</TabsTrigger>
            </TabsList>

            <TabsContent value="meetups">
              <Card className="p-6">
                <h3 className="text-lg font-bold mb-4">Meetup Funnel</h3>
                <div className="space-y-4">
                  <FunnelStep
                    label="Total Matches"
                    value={details?.totalMatches || 0}
                    percentage={100}
                  />
                  <FunnelStep
                    label="Unique Match Chats"
                    value={details?.uniqueMatchChats || 0}
                    percentage={
                      details?.totalMatches
                        ? (details.uniqueMatchChats / details.totalMatches) * 100
                        : 0
                    }
                  />
                  <FunnelStep
                    label="Meetups Proposed"
                    value={details?.meetupsProposed || 0}
                    percentage={
                      details?.uniqueMatchChats
                        ? (details.meetupsProposed / details.uniqueMatchChats) * 100
                        : 0
                    }
                  />
                  <FunnelStep
                    label="Meetups Confirmed"
                    value={details?.meetupsConfirmed || 0}
                    percentage={
                      details?.meetupsProposed
                        ? (details.meetupsConfirmed / details.meetupsProposed) * 100
                        : 0
                    }
                    highlight
                  />
                  <FunnelStep
                    label="Meetups Completed"
                    value={details?.meetupsCompleted || 0}
                    percentage={
                      details?.meetupsConfirmed
                        ? (details.meetupsCompleted / details.meetupsConfirmed) * 100
                        : 0
                    }
                    highlight
                  />
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="services">
              <Card className="p-6">
                <h3 className="text-lg font-bold mb-4">Service Intent Tracking</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                    <div>
                      <p className="font-semibold">Total Service Intents</p>
                      <p className="text-sm text-muted-foreground">All tracked interactions</p>
                    </div>
                    <div className="text-2xl font-bold">{details?.serviceIntents.total || 0}</div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                    <div>
                      <p className="font-semibold">Tap-to-Book Actions</p>
                      <p className="text-sm text-muted-foreground">Users who clicked "Book"</p>
                    </div>
                    <div className="text-2xl font-bold text-accent">
                      {details?.serviceIntents.tapBook || 0}
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-accent/10 rounded-lg border border-accent/20">
                    <div>
                      <p className="font-semibold">Confirmed Conversions</p>
                      <p className="text-sm text-muted-foreground">Users who actually booked (24h follow-up)</p>
                    </div>
                    <div className="text-2xl font-bold text-accent">
                      {details?.serviceIntents.conversions || 0}
                    </div>
                  </div>
                  <div className="flex items-center justify-between p-4 bg-green-500/10 rounded-lg border border-green-500/20">
                    <div>
                      <p className="font-semibold">Conversion Rate</p>
                      <p className="text-sm text-muted-foreground">Conversions / Tap-to-Book</p>
                    </div>
                    <div className="text-2xl font-bold text-green-600">
                      {details?.serviceIntents.tapBook
                        ? ((details.serviceIntents.conversions / details.serviceIntents.tapBook) * 100).toFixed(1)
                        : 0}%
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="events">
              <Card className="p-6">
                <h3 className="text-lg font-bold mb-4">Event Feedback Quality</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                    <div>
                      <p className="font-semibold">Total Event Feedback</p>
                      <p className="text-sm text-muted-foreground">Unique feedback submissions</p>
                    </div>
                    <div className="text-2xl font-bold">{details?.eventsFeedback.total || 0}</div>
                  </div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-accent/10 rounded-lg text-center">
                      <div className="text-3xl font-bold text-accent">
                        {details?.eventsFeedback.avgVibeScore.toFixed(1) || '0.0'}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">Avg Vibe Score</p>
                    </div>
                    <div className="p-4 bg-accent/10 rounded-lg text-center">
                      <div className="text-3xl font-bold text-accent">
                        {details?.eventsFeedback.avgPetDensity.toFixed(1) || '0.0'}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">Avg Pet Density</p>
                    </div>
                    <div className="p-4 bg-accent/10 rounded-lg text-center">
                      <div className="text-3xl font-bold text-accent">
                        {details?.eventsFeedback.avgVenueQuality.toFixed(1) || '0.0'}
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">Avg Venue Quality</p>
                    </div>
                  </div>
                </div>
              </Card>
            </TabsContent>

            <TabsContent value="engagement">
              <Card className="p-6">
                <h3 className="text-lg font-bold mb-4">User Engagement</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                    <div>
                      <p className="font-semibold">Avg Feedback Per User</p>
                      <p className="text-sm text-muted-foreground">Meetup + event feedback</p>
                    </div>
                    <div className="text-2xl font-bold">
                      {details?.avgFeedbackPerUser.toFixed(1) || '0.0'}
                    </div>
                  </div>
                  <div className="p-4 bg-accent/10 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-2">Data Collection Health</p>
                    <div className="text-2xl font-bold text-accent mb-1">
                      {metrics?.dataYieldPerUser.toFixed(1) || '0'} interactions/user
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Higher is better - shows users providing valuable training data
                    </p>
                  </div>
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  trend?: number;
  description: string;
  icon: any;
  color: 'blue' | 'green' | 'purple' | 'orange';
}

function MetricCard({ title, value, trend, description, icon: Icon, color }: MetricCardProps) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-500/20',
    green: 'from-green-500/20 to-green-600/20 border-green-500/20',
    purple: 'from-purple-500/20 to-purple-600/20 border-purple-500/20',
    orange: 'from-orange-500/20 to-orange-600/20 border-orange-500/20',
  };

  const iconColorClasses = {
    blue: 'text-blue-600',
    green: 'text-green-600',
    purple: 'text-purple-600',
    orange: 'text-orange-600',
  };

  return (
    <Card className={`p-6 bg-gradient-to-br ${colorClasses[color]} border`}>
      <div className="flex items-start justify-between mb-3">
        <Icon className={`w-6 h-6 ${iconColorClasses[color]}`} />
        {trend !== undefined && (
          <Badge variant={trend >= 0 ? 'default' : 'destructive'} className="text-xs">
            {trend >= 0 ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
            {Math.abs(trend).toFixed(1)}%
          </Badge>
        )}
      </div>
      <div className="text-3xl font-bold mb-1">{value}</div>
      <div className="text-sm font-semibold mb-1">{title}</div>
      <div className="text-xs text-muted-foreground">{description}</div>
    </Card>
  );
}

interface StatCardProps {
  label: string;
  value: number;
  icon: any;
  highlight?: boolean;
}

function StatCard({ label, value, icon: Icon, highlight }: StatCardProps) {
  return (
    <Card className={`p-4 ${highlight ? 'border-accent/50 bg-accent/5' : ''}`}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">{label}</p>
      </div>
      <div className="text-2xl font-bold">{value.toLocaleString()}</div>
    </Card>
  );
}

interface FunnelStepProps {
  label: string;
  value: number;
  percentage: number;
  highlight?: boolean;
}

function FunnelStep({ label, value, percentage, highlight }: FunnelStepProps) {
  return (
    <div className={`p-4 rounded-lg ${highlight ? 'bg-accent/10 border border-accent/20' : 'bg-muted/50'}`}>
      <div className="flex items-center justify-between mb-2">
        <p className="font-semibold">{label}</p>
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground">{percentage.toFixed(1)}%</span>
          <span className="text-lg font-bold">{value}</span>
        </div>
      </div>
      <div className="w-full bg-muted rounded-full h-2">
        <div
          className={`h-2 rounded-full ${highlight ? 'bg-accent' : 'bg-primary'}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  );
}
