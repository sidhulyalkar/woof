import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { UserStats, Badge } from "@/lib/types"

interface GamificationState {
  userStats: UserStats | null
  setUserStats: (stats: UserStats) => void
  addPoints: (points: number, reason: string) => void
  unlockBadge: (badge: Badge) => void
  incrementStreak: () => void
}

export const useGamificationStore = create<GamificationState>()(
  persist(
    (set) => ({
      userStats: null,
      setUserStats: (stats) => set({ userStats: stats }),
      addPoints: (points, reason) =>
        set((state) => {
          if (!state.userStats) return state
          const newPoints = state.userStats.points + points
          const newLevel = Math.floor(newPoints / 1000) + 1
          console.log(`[v0] Added ${points} points for: ${reason}`)
          return {
            userStats: {
              ...state.userStats,
              points: newPoints,
              level: newLevel,
            },
          }
        }),
      unlockBadge: (badge) =>
        set((state) => {
          if (!state.userStats) return state
          return {
            userStats: {
              ...state.userStats,
              badges: [...state.userStats.badges, badge],
            },
          }
        }),
      incrementStreak: () =>
        set((state) => {
          if (!state.userStats) return state
          return {
            userStats: {
              ...state.userStats,
              streaks: {
                ...state.userStats.streaks,
                daily: state.userStats.streaks.daily + 1,
              },
            },
          }
        }),
    }),
    {
      name: "petpath-gamification",
    },
  ),
)
