"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { BottomNav } from "@/components/bottom-nav"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Settings, Edit, MapPin, Calendar, Heart, MessageCircle, Briefcase, ChevronRight, LogOut, Loader2 } from "lucide-react"
import Link from "next/link"
import { EditProfileSheet } from "@/components/profile/edit-profile-sheet"
import { ServiceProviderCard } from "@/components/profile/service-provider-card"
import { LevelProgress } from "@/components/gamification/level-progress"
import { BadgeShowcase } from "@/components/gamification/badge-showcase"
import { useGamificationStore } from "@/lib/stores/gamification-store"
import { useSessionStore } from "@/store/session"
import { useQuery } from "@tanstack/react-query"
import { authApi } from "@/lib/api"
import { toast } from "sonner"

// Mock user data as fallback
const mockUser = {
  id: "current-user",
  name: "Alex Thompson",
  handle: "alex_t",
  email: "alex@example.com",
  age: 29,
  location: { lat: 37.7749, lng: -122.4194, address: "San Francisco, CA" },
  bio: "Dog lover and outdoor enthusiast. Always looking for new trails and dog-friendly adventures!",
  avatarUrl: "/user-avatar.jpg",
  createdAt: "2024-01-15",
  stats: {
    matches: 24,
    events: 12,
    activities: 48,
  },
  preferences: {
    activityLevel: "high",
    schedule: ["Weekday mornings", "Weekends"],
    interests: ["Hiking", "Dog parks", "Training"],
  },
  pets: [
    {
      id: "p1",
      name: "Charlie",
      species: "dog",
      breed: "Border Collie",
      age: 3,
      size: "medium",
      temperament: ["Energetic", "Friendly", "Smart"],
      photoUrl: "/border-collie.jpg",
    },
  ],
  isServiceProvider: true,
  services: {
    type: "dog-walking",
    rating: 4.8,
    reviewCount: 32,
    priceRange: "$25-40/hr",
  },
}

export default function ProfilePage() {
  const router = useRouter()
  const { user: sessionUser, clearSession } = useSessionStore()
  const [editOpen, setEditOpen] = useState(false)
  const { userStats, setUserStats } = useGamificationStore()

  // Fetch fresh user data
  const { data: userData, isLoading } = useQuery({
    queryKey: ['profile'],
    queryFn: authApi.me,
    enabled: !!sessionUser,
  })

  const user = userData?.user || sessionUser || mockUser

  const handleLogout = () => {
    clearSession()
    localStorage.removeItem('authToken')
    toast.success('Logged out successfully')
    router.replace('/login')
  }

  useEffect(() => {
    if (!userStats) {
      setUserStats({
        userId: user.id,
        points: 8450,
        level: 9,
        rank: 12,
        badges: [
          {
            id: "b1",
            name: "Early Bird",
            description: "Complete 10 morning walks",
            iconUrl: "/badge-early-bird.png",
            category: "activity",
            rarity: "common",
            unlockedAt: "2024-02-01",
          },
          {
            id: "b2",
            name: "Social Butterfly",
            description: "Attend 5 community events",
            iconUrl: "/badge-social.png",
            category: "social",
            rarity: "rare",
            unlockedAt: "2024-02-15",
          },
          {
            id: "b3",
            name: "Marathon Walker",
            description: "Walk 100 miles total",
            iconUrl: "/badge-marathon.png",
            category: "activity",
            rarity: "epic",
            unlockedAt: "2024-03-01",
          },
        ],
        streaks: {
          daily: 7,
          weekly: 3,
        },
        achievements: {
          totalWalks: 48,
          totalDistance: 125.5,
          totalEvents: 12,
          totalFriends: 24,
          totalPosts: 36,
        },
      })
    }
  }, [userStats, setUserStats, user.id])

  if (isLoading && !sessionUser) {
    return (
      <div className="min-h-screen flex items-center justify-center pb-20">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
        <BottomNav />
      </div>
    )
  }

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="px-4 py-4 max-w-lg mx-auto">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Profile</h1>
            <Button variant="ghost" size="icon" asChild>
              <Link href="/settings">
                <Settings className="w-5 h-5" />
              </Link>
            </Button>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="px-4 py-6 max-w-lg mx-auto space-y-6">
        {/* Profile Header */}
        <Card className="glass p-6 space-y-4">
          <div className="flex items-start gap-4">
            <Avatar className="w-20 h-20 border-2 border-border">
              <AvatarImage src={user.avatarUrl || "/placeholder.svg"} />
              <AvatarFallback>{(user.name || user.handle || 'U')[0].toUpperCase()}</AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <h2 className="text-xl font-bold mb-1">{user.name || user.handle}</h2>
              {user.email && (
                <div className="text-sm text-muted-foreground mb-2">
                  {user.email}
                </div>
              )}
              {user.location?.address && (
                <div className="flex items-center gap-1 text-sm text-muted-foreground mb-2">
                  <MapPin className="w-4 h-4" />
                  <span>{user.location.address}</span>
                </div>
              )}
              {user.createdAt && (
                <div className="flex items-center gap-1 text-sm text-muted-foreground">
                  <Calendar className="w-4 h-4" />
                  <span>
                    Joined {new Date(user.createdAt).toLocaleDateString("en-US", { month: "long", year: "numeric" })}
                  </span>
                </div>
              )}
            </div>
          </div>

          {user.bio && <p className="text-sm text-muted-foreground">{user.bio}</p>}

          <Button className="w-full gap-2" onClick={() => setEditOpen(true)}>
            <Edit className="w-4 h-4" />
            Edit Profile
          </Button>
        </Card>

        {userStats && (
          <Card className="glass p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Progress</h3>
              <Link href="/leaderboard">
                <Button variant="ghost" size="sm" className="text-accent">
                  Rank #{userStats.rank}
                </Button>
              </Link>
            </div>

            <LevelProgress level={userStats.level} points={userStats.points} />

            <div className="flex items-center justify-between pt-2">
              <div>
                <p className="text-sm text-muted-foreground">Badges Earned</p>
                <p className="text-2xl font-bold">{userStats.badges.length}</p>
              </div>
              <BadgeShowcase badges={userStats.badges} maxDisplay={3} />
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border/50">
              <div>
                <p className="text-sm text-muted-foreground">Daily Streak</p>
                <p className="text-xl font-bold text-accent">{userStats.streaks.daily} days</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Points</p>
                <p className="text-xl font-bold text-accent">{userStats.points.toLocaleString()}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Stats */}
        <div className="grid grid-cols-3 gap-3">
          <Card className="glass p-4 text-center">
            <Heart className="w-5 h-5 text-primary mx-auto mb-2" />
            <p className="text-2xl font-bold">{user.stats?.matches || 0}</p>
            <p className="text-xs text-muted-foreground">Matches</p>
          </Card>

          <Card className="glass p-4 text-center">
            <Calendar className="w-5 h-5 text-secondary mx-auto mb-2" />
            <p className="text-2xl font-bold">{user.stats?.events || 0}</p>
            <p className="text-xs text-muted-foreground">Events</p>
          </Card>

          <Card className="glass p-4 text-center">
            <MessageCircle className="w-5 h-5 text-accent mx-auto mb-2" />
            <p className="text-2xl font-bold">{user.stats?.activities || 0}</p>
            <p className="text-xs text-muted-foreground">Activities</p>
          </Card>
        </div>

        {/* Pets */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">My Pets</h3>
            <Button variant="ghost" size="sm" className="gap-2">
              <Edit className="w-4 h-4" />
              Manage
            </Button>
          </div>

          {user.pets && user.pets.length > 0 ? (
            user.pets.map((pet) => (
              <Card key={pet.id} className="glass p-4">
                <div className="flex items-start gap-4">
                  <Avatar className="w-16 h-16 border-2 border-border">
                    <AvatarImage src={pet.photoUrl || "/placeholder.svg"} />
                    <AvatarFallback>{pet.name[0]}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-semibold text-lg mb-1">{pet.name}</h4>
                    <p className="text-sm text-muted-foreground mb-2">
                      {pet.breed} • {pet.age} years {pet.size ? `• ${pet.size}` : ''}
                    </p>
                    {pet.temperament && (
                      <div className="flex flex-wrap gap-1">
                        {pet.temperament.map((trait) => (
                          <Badge key={trait} variant="outline" className="text-xs">
                            {trait}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">No pets added yet</p>
          )}
        </div>

        {/* Service Provider Section */}
        {user.isServiceProvider && user.services && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Service Provider</h3>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/services/manage" className="gap-2">
                  <Edit className="w-4 h-4" />
                  Manage
                </Link>
              </Button>
            </div>

            <ServiceProviderCard
              type={user.services.type}
              rating={user.services.rating}
              reviewCount={user.services.reviewCount}
              priceRange={user.services.priceRange}
            />
          </div>
        )}

        {/* Preferences */}
        {user.preferences && (
          <div className="space-y-3">
            <h3 className="text-lg font-semibold">Preferences</h3>
            <Card className="glass p-4 space-y-3">
              {user.preferences.activityLevel && (
                <div>
                  <p className="text-sm font-medium mb-2">Activity Level</p>
                  <Badge className="capitalize">{user.preferences.activityLevel}</Badge>
                </div>
              )}

              {user.preferences.schedule && user.preferences.schedule.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2">Availability</p>
                  <div className="flex flex-wrap gap-2">
                    {user.preferences.schedule.map((time) => (
                      <Badge key={time} variant="outline">
                        {time}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {user.preferences.interests && user.preferences.interests.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2">Interests</p>
                  <div className="flex flex-wrap gap-2">
                    {user.preferences.interests.map((interest) => (
                      <Badge key={interest} variant="outline">
                        {interest}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          </div>
        )}

        {/* Quick Actions */}
        <div className="space-y-2">
          <Button variant="outline" className="w-full justify-between bg-transparent" asChild>
            <Link href="/services">
              <div className="flex items-center gap-2">
                <Briefcase className="w-5 h-5" />
                <span>Browse Services</span>
              </div>
              <ChevronRight className="w-5 h-5" />
            </Link>
          </Button>

          <Button variant="outline" className="w-full justify-between bg-transparent" asChild>
            <Link href="/settings">
              <div className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                <span>Settings</span>
              </div>
              <ChevronRight className="w-5 h-5" />
            </Link>
          </Button>

          <Button
            variant="outline"
            className="w-full justify-start gap-2 text-destructive hover:text-destructive bg-transparent"
            onClick={handleLogout}
          >
            <LogOut className="w-5 h-5" />
            <span>Log Out</span>
          </Button>
        </div>
      </main>

      <EditProfileSheet open={editOpen} onOpenChange={setEditOpen} user={user} />
      <BottomNav />
    </div>
  )
}
