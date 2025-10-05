'use client';

import { useEffect, useState } from 'react';
import { trainingDataCollector, featureEngineer } from '@/lib/ml/trainingDataCollector';
import { Download, TrendingUp, Users, Heart, Calendar, Star, Database } from 'lucide-react';

export function MLAnalyticsDashboard() {
  const [analytics, setAnalytics] = useState({
    totalInteractions: 0,
    likeRate: 0,
    matchRate: 0,
    meetupCompletionRate: 0,
    avgMeetupRating: 0,
  });

  const [exportData, setExportData] = useState<any>(null);

  useEffect(() => {
    refreshAnalytics();
  }, []);

  const refreshAnalytics = () => {
    const data = trainingDataCollector.getAnalytics();
    setAnalytics(data);

    const exported = trainingDataCollector.exportTrainingData();
    setExportData(exported);
  };

  const handleExportCSV = () => {
    const dataPoints = trainingDataCollector.getStoredData();
    const csv = featureEngineer.exportToCSV(dataPoints);

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `petpath-ml-training-data-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const exported = trainingDataCollector.exportTrainingData();
    const json = JSON.stringify(exported, null, 2);

    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `petpath-ml-training-data-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleClearData = () => {
    if (confirm('Are you sure you want to clear all training data? This cannot be undone.')) {
      trainingDataCollector.clearData();
      refreshAnalytics();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">ML Analytics Dashboard</h1>
          <p className="text-gray-600">
            Track user interactions and training data collection for ML model development
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <StatCard
            icon={<Database className="h-6 w-6 text-blue-600" />}
            title="Total Interactions"
            value={analytics.totalInteractions}
            subtitle="Data points collected"
            color="blue"
          />

          <StatCard
            icon={<Heart className="h-6 w-6 text-pink-600" />}
            title="Like Rate"
            value={`${analytics.likeRate.toFixed(1)}%`}
            subtitle="Of profiles liked"
            color="pink"
          />

          <StatCard
            icon={<Users className="h-6 w-6 text-purple-600" />}
            title="Match Rate"
            value={`${analytics.matchRate.toFixed(1)}%`}
            subtitle="Successful matches"
            color="purple"
          />

          <StatCard
            icon={<Calendar className="h-6 w-6 text-green-600" />}
            title="Meetup Completion"
            value={`${analytics.meetupCompletionRate.toFixed(1)}%`}
            subtitle="Of matches"
            color="green"
          />

          <StatCard
            icon={<Star className="h-6 w-6 text-yellow-600" />}
            title="Avg Meetup Rating"
            value={analytics.avgMeetupRating.toFixed(1)}
            subtitle="Out of 5.0"
            color="yellow"
          />

          <StatCard
            icon={<TrendingUp className="h-6 w-6 text-indigo-600" />}
            title="Conversion Funnel"
            value={`${((analytics.meetupCompletionRate / 100) * (analytics.matchRate / 100) * 100).toFixed(1)}%`}
            subtitle="Like → Meetup"
            color="indigo"
          />
        </div>

        {/* Export Data Section */}
        <div className="bg-white rounded-3xl shadow-lg p-8 border border-gray-100 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Export Training Data</h2>
          <p className="text-gray-600 mb-6">
            Download collected data for ML model training or investor presentations
          </p>

          {exportData && (
            <div className="bg-gray-50 rounded-2xl p-6 mb-6">
              <h3 className="font-semibold text-gray-900 mb-3">Dataset Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="text-gray-600">Total Samples</div>
                  <div className="font-bold text-lg text-gray-900">{exportData.metadata.totalSamples}</div>
                </div>
                <div>
                  <div className="text-gray-600">Likes</div>
                  <div className="font-bold text-lg text-pink-600">{exportData.metadata.likeCount}</div>
                </div>
                <div>
                  <div className="text-gray-600">Matches</div>
                  <div className="font-bold text-lg text-purple-600">{exportData.metadata.matchCount}</div>
                </div>
                <div>
                  <div className="text-gray-600">Meetups</div>
                  <div className="font-bold text-lg text-green-600">{exportData.metadata.meetupCount}</div>
                </div>
                <div>
                  <div className="text-gray-600">Avg Rating</div>
                  <div className="font-bold text-lg text-yellow-600">
                    {exportData.metadata.avgMeetupRating.toFixed(2)} / 5.0
                  </div>
                </div>
                <div>
                  <div className="text-gray-600">Export Date</div>
                  <div className="font-bold text-sm text-gray-900">
                    {new Date(exportData.metadata.exportDate).toLocaleDateString()}
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              onClick={handleExportCSV}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-full hover:from-blue-600 hover:to-blue-700 font-semibold transition-all"
            >
              <Download className="h-5 w-5" />
              Export CSV (for Python/R)
            </button>

            <button
              onClick={handleExportJSON}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-full hover:from-purple-600 hover:to-purple-700 font-semibold transition-all"
            >
              <Download className="h-5 w-5" />
              Export JSON
            </button>

            <button
              onClick={handleClearData}
              className="flex items-center gap-2 px-6 py-3 border-2 border-red-300 text-red-600 rounded-full hover:border-red-400 hover:bg-red-50 font-semibold transition-all ml-auto"
            >
              Clear All Data
            </button>
          </div>
        </div>

        {/* ML Readiness Indicator */}
        <div className="bg-white rounded-3xl shadow-lg p-8 border border-gray-100">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">ML Model Readiness</h2>

          <MLReadinessIndicator
            totalInteractions={analytics.totalInteractions}
            meetupCount={exportData?.metadata.meetupCount || 0}
          />
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  subtitle: string;
  color: string;
}

function StatCard({ icon, title, value, subtitle, color }: StatCardProps) {
  const colorClasses = {
    blue: 'from-blue-50 to-blue-100 border-blue-200',
    pink: 'from-pink-50 to-pink-100 border-pink-200',
    purple: 'from-purple-50 to-purple-100 border-purple-200',
    green: 'from-green-50 to-green-100 border-green-200',
    yellow: 'from-yellow-50 to-yellow-100 border-yellow-200',
    indigo: 'from-indigo-50 to-indigo-100 border-indigo-200',
  };

  return (
    <div
      className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue} rounded-2xl p-6 border`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="p-2 bg-white rounded-lg shadow-sm">{icon}</div>
      </div>
      <div className="text-3xl font-bold text-gray-900 mb-1">{value}</div>
      <div className="text-sm font-semibold text-gray-700 mb-1">{title}</div>
      <div className="text-xs text-gray-600">{subtitle}</div>
    </div>
  );
}

function MLReadinessIndicator({ totalInteractions, meetupCount }: { totalInteractions: number; meetupCount: number }) {
  const targets = {
    minInteractions: 500,
    targetInteractions: 1000,
    minMeetups: 30,
    targetMeetups: 100,
  };

  const interactionProgress = Math.min((totalInteractions / targets.targetInteractions) * 100, 100);
  const meetupProgress = Math.min((meetupCount / targets.targetMeetups) * 100, 100);

  const isReady = totalInteractions >= targets.minInteractions && meetupCount >= targets.minMeetups;

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-gray-900">Interaction Data</span>
          <span className="text-sm text-gray-600">
            {totalInteractions} / {targets.targetInteractions}
          </span>
        </div>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
            style={{ width: `${interactionProgress}%` }}
          />
        </div>
        {totalInteractions < targets.minInteractions && (
          <div className="text-xs text-gray-600 mt-1">
            Need {targets.minInteractions - totalInteractions} more interactions for minimum viable dataset
          </div>
        )}
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="font-semibold text-gray-900">Meetup Ratings</span>
          <span className="text-sm text-gray-600">
            {meetupCount} / {targets.targetMeetups}
          </span>
        </div>
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-500"
            style={{ width: `${meetupProgress}%` }}
          />
        </div>
        {meetupCount < targets.minMeetups && (
          <div className="text-xs text-gray-600 mt-1">
            Need {targets.minMeetups - meetupCount} more rated meetups for quality labels
          </div>
        )}
      </div>

      <div
        className={`p-4 rounded-2xl ${isReady ? 'bg-green-50 border-2 border-green-200' : 'bg-yellow-50 border-2 border-yellow-200'}`}
      >
        <div className="flex items-start gap-3">
          <div className="text-2xl">{isReady ? '✅' : '⏳'}</div>
          <div className="flex-1">
            <div className="font-bold text-gray-900 mb-1">
              {isReady ? 'Ready for ML Model Training' : 'Collecting Training Data'}
            </div>
            <div className="text-sm text-gray-700">
              {isReady
                ? 'You have enough data to begin training a machine learning model. Export the data and follow the ML_SYSTEM_README.md guide.'
                : 'Continue collecting user interactions and meetup ratings. This data will be used to train a personalized matching algorithm.'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
