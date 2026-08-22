/**
 * Goals API Client
 *
 * Handles all goal-related API calls for the mobile app
 */

import apiClient from './client';

export enum GoalType {
  DISTANCE = 'DISTANCE',
  TIME = 'TIME',
  STEPS = 'STEPS',
  ACTIVITIES = 'ACTIVITIES',
  CALORIES = 'CALORIES',
  SOCIAL = 'SOCIAL',
}

export enum GoalPeriod {
  DAILY = 'DAILY',
  WEEKLY = 'WEEKLY',
  MONTHLY = 'MONTHLY',
  CUSTOM = 'CUSTOM',
}

export enum GoalStatus {
  ACTIVE = 'ACTIVE',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  PAUSED = 'PAUSED',
}

export interface Goal {
  id: string;
  userId: string;
  petId: string;
  goalType: GoalType;
  period: GoalPeriod;
  targetNumber: number;
  targetUnit: string;
  progress: number;
  currentValue: number;
  status: GoalStatus;
  startDate: string;
  endDate: string;
  reminderTime?: string;
  isRecurring: boolean;
  streakCount: number;
  bestStreak: number;
  completedDays: string[];
  metadata?: unknown;
  createdAt: string;
  updatedAt: string;
  pet?: {
    id: string;
    name: string;
    avatarUrl?: string;
  };
}

export interface CreateGoalRequest {
  petId: string;
  goalType: GoalType;
  period: GoalPeriod;
  targetNumber: number;
  targetUnit: string;
  startDate: string;
  endDate: string;
  reminderTime?: string;
  isRecurring?: boolean;
  metadata?: unknown;
}

export interface UpdateGoalRequest {
  targetNumber?: number;
  targetUnit?: string;
  status?: GoalStatus;
  reminderTime?: string;
  metadata?: unknown;
}

export interface GoalStatistics {
  totalGoals: number;
  activeGoals: number;
  completedGoals: number;
  failedGoals: number;
  averageProgress: number;
  longestStreak: number;
  currentStreak: number;
}

const goalsApi = {
  /**
   * Get all goals for the current user
   */
  async getGoals(petId?: string, status?: GoalStatus): Promise<Goal[]> {
    const params: Record<string, string> = {};
    if (petId) params.petId = petId;
    if (status) params.status = status;

    return apiClient.get<Goal[]>('/goals', { params });
  },

  /**
   * Get a single goal by ID
   */
  async getGoal(id: string): Promise<Goal> {
    return apiClient.get<Goal>(`/goals/${id}`);
  },

  /**
   * Create a new goal
   */
  async createGoal(data: CreateGoalRequest): Promise<Goal> {
    return apiClient.post<Goal>('/goals', data);
  },

  /**
   * Update a goal
   */
  async updateGoal(id: string, data: UpdateGoalRequest): Promise<Goal> {
    return apiClient.patch<Goal>(`/goals/${id}`, data);
  },

  /**
   * Update goal progress
   */
  async updateProgress(id: string, value: number): Promise<Goal> {
    return apiClient.patch<Goal>(`/goals/${id}/progress`, { value });
  },

  /**
   * Delete a goal
   */
  async deleteGoal(id: string): Promise<void> {
    await apiClient.delete(`/goals/${id}`);
  },

  /**
   * Get goal statistics for the current user
   */
  async getStatistics(): Promise<GoalStatistics> {
    return apiClient.get<GoalStatistics>('/goals/statistics');
  },

  /**
   * Helper: Calculate days remaining
   */
  getDaysRemaining(goal: Goal): number {
    const now = new Date();
    const end = new Date(goal.endDate);
    const diff = end.getTime() - now.getTime();
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
  },

  /**
   * Helper: Check if goal is completed today
   */
  isCompletedToday(goal: Goal): boolean {
    const today = new Date().toISOString().split('T')[0];
    return goal.completedDays.includes(today);
  },

  /**
   * Helper: Get goal icon name
   */
  getGoalIcon(goalType: GoalType): string {
    const icons: Record<GoalType, string> = {
      [GoalType.DISTANCE]: 'walk',
      [GoalType.TIME]: 'time',
      [GoalType.STEPS]: 'footsteps',
      [GoalType.ACTIVITIES]: 'fitness',
      [GoalType.CALORIES]: 'flame',
      [GoalType.SOCIAL]: 'people',
    };
    return icons[goalType] || 'flag';
  },

  /**
   * Helper: Get goal color
   */
  getGoalColor(goalType: GoalType): string {
    const colors: Record<GoalType, string> = {
      [GoalType.DISTANCE]: '#10b981',
      [GoalType.TIME]: '#3b82f6',
      [GoalType.STEPS]: '#8b5cf6',
      [GoalType.ACTIVITIES]: '#f59e0b',
      [GoalType.CALORIES]: '#ef4444',
      [GoalType.SOCIAL]: '#ec4899',
    };
    return colors[goalType] || '#6b7280';
  },
};

export default goalsApi;
