/**
 * Goals Screen
 *
 * Displays and manages fitness goals for pets with:
 * - Progress tracking with animated progress bars
 * - Streak visualization with fire icons
 * - Quick stats cards
 * - Goal creation and editing
 * - Completion celebrations
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  Alert,
  Animated,
  Dimensions,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import goalsApi, { Goal, GoalType, GoalPeriod, GoalStatistics } from '../api/goals';

const { width } = Dimensions.get('window');

const GoalsScreen = ({ navigation }: any) => {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [statistics, setStatistics] = useState<GoalStatistics | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'active' | 'completed'>('active');

  // Load goals and statistics
  const loadData = async () => {
    try {
      const [goalsData, statsData] = await Promise.all([
        goalsApi.getGoals(undefined, selectedFilter === 'all' ? undefined : selectedFilter.toUpperCase() as any),
        goalsApi.getStatistics(),
      ]);
      setGoals(goalsData);
      setStatistics(statsData);
    } catch (error) {
      console.error('Failed to load goals:', error);
      Alert.alert('Error', 'Failed to load goals');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedFilter]);

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const handleCreateGoal = () => {
    navigation.navigate('CreateGoal');
  };

  const handleGoalPress = (goal: Goal) => {
    navigation.navigate('GoalDetails', { goalId: goal.id });
  };

  const getProgressColor = (progress: number): string => {
    if (progress >= 100) return '#10b981'; // Green
    if (progress >= 75) return '#3b82f6'; // Blue
    if (progress >= 50) return '#f59e0b'; // Orange
    if (progress >= 25) return '#ef4444'; // Red
    return '#6b7280'; // Gray
  };

  const renderStatCard = (icon: string, label: string, value: string | number, color: string) => (
    <View style={[styles.statCard, { borderLeftColor: color }]}>
      <Ionicons name={icon as any} size={24} color={color} />
      <View style={styles.statContent}>
        <Text style={styles.statLabel}>{label}</Text>
        <Text style={styles.statValue}>{value}</Text>
      </View>
    </View>
  );

  const renderGoalCard = (goal: Goal) => {
    const progress = Math.min(goal.progress, 100);
    const icon = goalsApi.getGoalIcon(goal.goalType);
    const color = goalsApi.getGoalColor(goal.goalType);
    const daysRemaining = goalsApi.getDaysRemaining(goal);
    const isCompletedToday = goalsApi.isCompletedToday(goal);

    return (
      <TouchableOpacity
        key={goal.id}
        style={styles.goalCard}
        onPress={() => handleGoalPress(goal)}
        activeOpacity={0.7}
      >
        {/* Header */}
        <View style={styles.goalHeader}>
          <View style={styles.goalTitleRow}>
            <View style={[styles.iconBadge, { backgroundColor: color + '20' }]}>
              <Ionicons name={icon as any} size={20} color={color} />
            </View>
            <View style={styles.goalTitleContainer}>
              <Text style={styles.goalTitle}>
                {goal.goalType.charAt(0) + goal.goalType.slice(1).toLowerCase()}
              </Text>
              <Text style={styles.goalPet}>{goal.pet?.name || 'Unknown Pet'}</Text>
            </View>
          </View>

          {/* Streak Badge */}
          {goal.streakCount > 0 && (
            <View style={styles.streakBadge}>
              <Ionicons name="flame" size={16} color="#f59e0b" />
              <Text style={styles.streakText}>{goal.streakCount}</Text>
            </View>
          )}
        </View>

        {/* Progress */}
        <View style={styles.progressSection}>
          <View style={styles.progressHeader}>
            <Text style={styles.progressText}>
              {goal.currentValue.toFixed(1)} / {goal.targetNumber} {goal.targetUnit}
            </Text>
            <Text style={[styles.progressPercentage, { color }]}>
              {progress.toFixed(0)}%
            </Text>
          </View>

          {/* Progress Bar */}
          <View style={styles.progressBarContainer}>
            <View
              style={[
                styles.progressBarFill,
                {
                  width: `${progress}%`,
                  backgroundColor: color,
                },
              ]}
            />
          </View>
        </View>

        {/* Footer */}
        <View style={styles.goalFooter}>
          <View style={styles.footerItem}>
            <Ionicons name="calendar-outline" size={14} color="#6b7280" />
            <Text style={styles.footerText}>
              {daysRemaining > 0 ? `${daysRemaining} days left` : 'Expired'}
            </Text>
          </View>

          {isCompletedToday && (
            <View style={styles.completedBadge}>
              <Ionicons name="checkmark-circle" size={14} color="#10b981" />
              <Text style={styles.completedText}>Today ✓</Text>
            </View>
          )}

          {goal.status === 'COMPLETED' && (
            <View style={styles.statusBadge}>
              <Text style={styles.statusText}>Completed</Text>
            </View>
          )}
        </View>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <Text>Loading goals...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Goals</Text>
        <TouchableOpacity onPress={handleCreateGoal} style={styles.addButton}>
          <Ionicons name="add-circle" size={28} color="#8b5cf6" />
        </TouchableOpacity>
      </View>

      <ScrollView
        style={styles.scrollView}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
        showsVerticalScrollIndicator={false}
      >
        {/* Statistics Cards */}
        {statistics && (
          <View style={styles.statsContainer}>
            {renderStatCard('trophy', 'Active', statistics.activeGoals, '#8b5cf6')}
            {renderStatCard('checkmark-done', 'Completed', statistics.completedGoals, '#10b981')}
            {renderStatCard('flame', 'Current Streak', statistics.currentStreak, '#f59e0b')}
            {renderStatCard('trending-up', 'Avg Progress', `${statistics.averageProgress}%`, '#3b82f6')}
          </View>
        )}

        {/* Filter Tabs */}
        <View style={styles.filterContainer}>
          <TouchableOpacity
            style={[styles.filterTab, selectedFilter === 'active' && styles.filterTabActive]}
            onPress={() => setSelectedFilter('active')}
          >
            <Text style={[styles.filterText, selectedFilter === 'active' && styles.filterTextActive]}>
              Active
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.filterTab, selectedFilter === 'all' && styles.filterTabActive]}
            onPress={() => setSelectedFilter('all')}
          >
            <Text style={[styles.filterText, selectedFilter === 'all' && styles.filterTextActive]}>
              All
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.filterTab, selectedFilter === 'completed' && styles.filterTabActive]}
            onPress={() => setSelectedFilter('completed')}
          >
            <Text style={[styles.filterText, selectedFilter === 'completed' && styles.filterTextActive]}>
              Completed
            </Text>
          </TouchableOpacity>
        </View>

        {/* Goals List */}
        <View style={styles.goalsContainer}>
          {goals.length === 0 ? (
            <View style={styles.emptyState}>
              <Ionicons name="flag-outline" size={64} color="#d1d5db" />
              <Text style={styles.emptyTitle}>No goals yet</Text>
              <Text style={styles.emptyText}>
                Create your first goal to start tracking progress!
              </Text>
              <TouchableOpacity style={styles.createButton} onPress={handleCreateGoal}>
                <Text style={styles.createButtonText}>Create Goal</Text>
              </TouchableOpacity>
            </View>
          ) : (
            goals.map(renderGoalCard)
          )}
        </View>
      </ScrollView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9fafb',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 60,
    paddingBottom: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#e5e7eb',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#111827',
  },
  addButton: {
    padding: 4,
  },
  scrollView: {
    flex: 1,
  },
  statsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    padding: 16,
    gap: 12,
  },
  statCard: {
    flex: 1,
    minWidth: (width - 52) / 2,
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    borderLeftWidth: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  statContent: {
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#111827',
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    gap: 8,
  },
  filterTab: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 20,
    backgroundColor: '#f3f4f6',
    alignItems: 'center',
  },
  filterTabActive: {
    backgroundColor: '#8b5cf6',
  },
  filterText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#6b7280',
  },
  filterTextActive: {
    color: '#fff',
  },
  goalsContainer: {
    padding: 16,
  },
  goalCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  goalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 12,
  },
  goalTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconBadge: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  goalTitleContainer: {
    flex: 1,
  },
  goalTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 2,
  },
  goalPet: {
    fontSize: 13,
    color: '#6b7280',
  },
  streakBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  streakText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#f59e0b',
  },
  progressSection: {
    marginBottom: 12,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressText: {
    fontSize: 14,
    color: '#111827',
    fontWeight: '500',
  },
  progressPercentage: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  goalFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  footerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  footerText: {
    fontSize: 12,
    color: '#6b7280',
  },
  completedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: '#d1fae5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  completedText: {
    fontSize: 12,
    color: '#10b981',
    fontWeight: '600',
  },
  statusBadge: {
    backgroundColor: '#dbeafe',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  statusText: {
    fontSize: 12,
    color: '#3b82f6',
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    fontSize: 14,
    color: '#6b7280',
    textAlign: 'center',
    marginBottom: 24,
    paddingHorizontal: 32,
  },
  createButton: {
    backgroundColor: '#8b5cf6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 24,
  },
  createButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default GoalsScreen;
