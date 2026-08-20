"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import {
  Activity,
  CalendarDays,
  ChevronRight,
  Edit,
  Image as ImageIcon,
  Loader2,
  LogOut,
  PawPrint,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
  Trophy,
} from "lucide-react"
import { toast } from "sonner"
import { BottomNav } from "@/components/bottom-nav"
import { EditProfileSheet } from "@/components/profile/edit-profile-sheet"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { authApi } from "@/lib/api"
import { profileApi } from "@/lib/api/profile"
import { quizApi } from "@/lib/api/quiz"
import type { AuthUser } from "@/lib/stores/auth-store"
import { useAuthStore } from "@/lib/stores/auth-store"

const visibilityLabel = {
  PUBLIC: "Public profile",
  FRIENDS_ONLY: "Friends only",
  PRIVATE: "Private profile",
} as const

export default function ProfilePage() {
  const router = useRouter()
  const queryClient = useQueryClient()
  const cachedUser = useAuthStore((state) => state.user)
  const [editOpen, setEditOpen] = useState(false)

  const {
    data: profile,
    isLoading,
    error,
    refetch,
  } = useQuery<AuthUser>({
    queryKey: ["auth-profile"],
    queryFn: authApi.me,
    staleTime: 30_000,
  })

  const user = profile ?? cachedUser

  const { data: gamification } = useQuery({
    queryKey: ["gamification-summary"],
    queryFn: profileApi.gamificationSummary,
    enabled: Boolean(user),
    staleTime: 60_000,
  })

  const { data: latestPreferences } = useQuery({
    queryKey: ["quiz-latest"],
    queryFn: quizApi.latest,
    enabled: Boolean(user),
    staleTime: 60_000,
  })

  const handleLogout = () => {
    authApi.logout()
    queryClient.clear()
    toast.success("Signed out")
    router.replace("/login")
  }

  const handleProfileSaved = (updated: AuthUser) => {
    queryClient.setQueryData(["auth-profile"], updated)
  }

  if (isLoading && !user) {
    return (
      <div className="flex min-h-screen items-center justify-center" role="status">
        <Loader2 className="h-8 w-8 animate-spin text-primary" aria-hidden="true" />
        <span className="sr-only">Loading profile</span>
      </div>
    )
  }

  if (!user) {
    return (
      <main id="main-content" className="mx-auto flex min-h-screen max-w-xl flex-col items-center justify-center px-6 text-center">
        <h1 className="text-xl font-semibold">Profile unavailable</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          {error ? "Woof could not refresh your account profile." : "No authenticated profile is available."}
        </p>
        <Button className="mt-5" onClick={() => refetch()}>
          Try again
        </Button>
      </main>
    )
  }

  const pets = user.pets ?? []
  const memberSince = user.createdAt
    ? new Intl.DateTimeFormat(undefined, { month: "short", year: "numeric" }).format(new Date(user.createdAt))
    : null

  const stats = [
    { label: "Activities", value: user._count?.activities ?? 0, icon: Activity },
    { label: "Posts", value: user._count?.posts ?? 0, icon: ImageIcon },
    { label: "Points", value: gamification?.points ?? user.totalPoints ?? user.points ?? 0, icon: Trophy },
  ]

  return (
    <div className="min-h-screen pb-24">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex h-16 max-w-xl items-center justify-between px-4">
          <div>
            <p className="eyebrow">Account & pack</p>
            <h1 className="mt-0.5 text-xl font-bold tracking-tight">Profile</h1>
          </div>
          <Button variant="ghost" size="icon" asChild className="rounded-xl">
            <Link href="/settings" aria-label="Open settings">
              <Settings className="h-5 w-5" aria-hidden="true" />
            </Link>
          </Button>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-xl space-y-5 px-4 py-5">
        <section className="glass rounded-3xl p-5 sm:p-6" aria-labelledby="profile-identity">
          <div className="flex items-start gap-4">
            <Avatar className="h-20 w-20 border-2 border-border sm:h-24 sm:w-24">
              <AvatarImage src={user.avatarUrl || "/placeholder.svg"} alt="" />
              <AvatarFallback className="text-xl font-bold">{user.handle.slice(0, 1).toUpperCase()}</AvatarFallback>
            </Avatar>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 id="profile-identity" className="truncate text-2xl font-bold tracking-tight">@{user.handle}</h2>
                {user.isVerified && (
                  <Badge className="border-secondary/20 bg-secondary/10 text-secondary hover:bg-secondary/10">
                    <ShieldCheck className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
                    Verified
                  </Badge>
                )}
              </div>
              <p className="mt-1 text-sm text-muted-foreground">{user.email}</p>
              <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
                <Badge variant="outline">{visibilityLabel[user.visibility ?? "PUBLIC"]}</Badge>
                {memberSince && (
                  <Badge variant="outline">
                    <CalendarDays className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
                    Joined {memberSince}
                  </Badge>
                )}
              </div>
            </div>
          </div>

          {user.bio ? (
            <p className="mt-5 text-sm leading-6 text-muted-foreground">{user.bio}</p>
          ) : (
            <p className="mt-5 text-sm italic text-muted-foreground">Add a short bio to give other owners context before they start a conversation.</p>
          )}

          <Button variant="outline" className="mt-5 w-full gap-2 bg-transparent" onClick={() => setEditOpen(true)}>
            <Edit className="h-4 w-4" aria-hidden="true" />
            Edit public profile
          </Button>
        </section>

        <section className="grid grid-cols-3 gap-3" aria-label="Profile stats">
          {stats.map((stat) => {
            const Icon = stat.icon
            return (
              <Card key={stat.label} className="surface-soft rounded-2xl p-4 text-center">
                <Icon className="mx-auto h-4 w-4 text-primary" aria-hidden="true" />
                <p className="mt-2 text-xl font-bold">{stat.value}</p>
                <p className="mt-0.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{stat.label}</p>
              </Card>
            )
          })}
        </section>

        <section aria-labelledby="pets-heading" className="space-y-3">
          <div className="flex items-end justify-between gap-3">
            <div>
              <p className="eyebrow">The pack</p>
              <h2 id="pets-heading" className="mt-1 text-lg font-bold">Pets</h2>
            </div>
            <span className="text-xs text-muted-foreground">{pets.length} saved</span>
          </div>

          {pets.length > 0 ? (
            <div className="space-y-3">
              {pets.map((pet) => (
                <Card key={pet.id} className="surface-soft flex items-center gap-4 rounded-2xl p-4">
                  <Avatar className="h-14 w-14 border border-border">
                    <AvatarImage src={pet.avatarUrl || "/placeholder.svg"} alt="" />
                    <AvatarFallback><PawPrint className="h-5 w-5" aria-hidden="true" /></AvatarFallback>
                  </Avatar>
                  <div className="min-w-0 flex-1">
                    <p className="font-semibold">{pet.name}</p>
                    <p className="mt-0.5 truncate text-sm capitalize text-muted-foreground">
                      {[pet.breed, pet.species].filter(Boolean).join(" · ")}
                    </p>
                  </div>
                  <Button variant="ghost" size="icon" asChild aria-label={`Open ${pet.name}'s profile`}>
                    <Link href={`/pets/${pet.id}`}>
                      <ChevronRight className="h-5 w-5" aria-hidden="true" />
                    </Link>
                  </Button>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="surface-soft rounded-2xl p-5 text-center">
              <PawPrint className="mx-auto h-6 w-6 text-primary" aria-hidden="true" />
              <p className="mt-3 font-semibold">No pet profile yet</p>
              <p className="mt-1 text-sm text-muted-foreground">Discovery stays disabled until matching has a pet to reason about.</p>
            </Card>
          )}
        </section>

        <section className="space-y-3" aria-labelledby="learning-heading">
          <div>
            <p className="eyebrow">Learning signals</p>
            <h2 id="learning-heading" className="mt-1 text-lg font-bold">Matching context</h2>
          </div>
          <Card className="surface-soft rounded-2xl p-4">
            <div className="flex items-start gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                <SlidersHorizontal className="h-5 w-5" aria-hidden="true" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-semibold">Preference session</p>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {latestPreferences?.completedAt
                    ? `Last captured ${new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(latestPreferences.completedAt))}. Preferences are stored separately from durable pet traits.`
                    : "No saved matching preference session yet. The deterministic baseline can still operate from pet profile data."}
                </p>
              </div>
            </div>
          </Card>
        </section>

        <section className="space-y-2" aria-label="Profile actions">
          <Button variant="outline" className="w-full justify-between bg-transparent" asChild>
            <Link href="/settings">
              <span>Privacy, notifications, and account settings</span>
              <ChevronRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          </Button>
          <Button variant="ghost" className="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive" onClick={handleLogout}>
            <LogOut className="h-4 w-4" aria-hidden="true" />
            Sign out
          </Button>
        </section>
      </main>

      <EditProfileSheet open={editOpen} onOpenChange={setEditOpen} user={user} onSaved={handleProfileSaved} />
      <BottomNav />
    </div>
  )
}
