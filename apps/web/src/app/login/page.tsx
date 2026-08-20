"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState } from "react"
import { CheckCircle2, Loader2, PawPrint, ShieldCheck, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { authApi } from "@/lib/api"

const productSignals = [
  "Explainable compatibility, not mystery scores",
  "Designed around real-world meetup outcomes",
  "Location and trust treated as product boundaries",
]

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setError("")
    setIsLoading(true)

    try {
      await authApi.login({ email, password })
      router.replace("/")
    } catch (err: any) {
      console.error("Login failed", err)
      setError(err?.response?.data?.message || "Invalid email or password.")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[1.05fr_0.95fr]">
      <aside className="relative hidden overflow-hidden border-r border-border/60 lg:flex lg:flex-col lg:justify-between lg:p-12 xl:p-16">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(255,180,84,0.14),transparent_32rem),radial-gradient(circle_at_85%_75%,rgba(85,214,190,0.12),transparent_28rem)]" aria-hidden="true" />
        <div className="relative">
          <Link href="/login" className="inline-flex min-h-0 min-w-0 items-center gap-3" aria-label="Woof sign in">
            <span className="brand-mark flex h-11 w-11 items-center justify-center rounded-2xl">
              <PawPrint className="h-6 w-6 text-primary-foreground" aria-hidden="true" />
            </span>
            <span className="text-xl font-bold tracking-tight">Woof</span>
          </Link>
        </div>

        <div className="relative max-w-xl">
          <p className="eyebrow">A social network that closes the loop</p>
          <h1 className="mt-4 text-4xl font-bold leading-tight tracking-tight xl:text-5xl">
            Better dog friendships start with better context.
          </h1>
          <p className="mt-5 max-w-lg text-base leading-7 text-muted-foreground">
            Discover compatible dogs, coordinate a safe meetup, and learn from what actually happened offline instead of optimizing another endless feed.
          </p>

          <div className="mt-8 space-y-3">
            {productSignals.map((signal) => (
              <div key={signal} className="flex items-center gap-3 text-sm text-foreground/90">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-secondary/10 text-secondary">
                  <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
                </span>
                {signal}
              </div>
            ))}
          </div>
        </div>

        <div className="relative flex items-center gap-2 text-xs text-muted-foreground">
          <ShieldCheck className="h-4 w-4 text-secondary" aria-hidden="true" />
          Precise location should stay contextual and permissioned.
        </div>
      </aside>

      <main id="main-content" className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6 lg:px-10">
        <div className="w-full max-w-md">
          <div className="mb-8 flex items-center gap-3 lg:hidden">
            <span className="brand-mark flex h-10 w-10 items-center justify-center rounded-xl">
              <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
            </span>
            <div>
              <p className="eyebrow">Your local pack</p>
              <p className="font-bold">Woof</p>
            </div>
          </div>

          <Card className="glass rounded-3xl border-border/60 shadow-2xl shadow-black/10">
            <CardHeader className="space-y-3 pb-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Sparkles className="h-5 w-5" aria-hidden="true" />
              </div>
              <div>
                <CardTitle className="text-3xl font-bold tracking-tight">Welcome back</CardTitle>
                <CardDescription className="mt-2 leading-relaxed">
                  Sign in to continue building your dog&apos;s real-world social graph.
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    autoComplete="email"
                    placeholder="you@example.com"
                    value={email}
                    onChange={(event) => setEmail(event.target.value)}
                    required
                    disabled={isLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <Input
                    id="password"
                    type="password"
                    autoComplete="current-password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    required
                    disabled={isLoading}
                  />
                </div>

                {error && (
                  <div role="alert" aria-live="polite" className="rounded-xl border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
                    {error}
                  </div>
                )}

                <Button type="submit" size="lg" className="w-full" disabled={isLoading}>
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                      Signing in…
                    </>
                  ) : (
                    "Sign in"
                  )}
                </Button>
              </form>

              <p className="mt-6 text-center text-sm text-muted-foreground">
                New to Woof?{" "}
                <Link href="/onboarding" className="font-semibold text-primary hover:text-primary/80">
                  Create your profile
                </Link>
              </p>
            </CardContent>
          </Card>

          <p className="mt-5 text-center text-xs leading-relaxed text-muted-foreground">
            Woof is a portfolio research product. Public demos should use synthetic data and avoid sensitive real-world location history.
          </p>
        </div>
      </main>
    </div>
  )
}
