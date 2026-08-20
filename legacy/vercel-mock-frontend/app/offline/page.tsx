"use client"

import { WifiOff } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function OfflinePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4">
      <WifiOff className="h-16 w-16 text-muted-foreground/50" />
      <h1 className="mt-6 text-2xl font-bold">You're Offline</h1>
      <p className="mt-2 text-center text-muted-foreground">
        It looks like you've lost your internet connection. Some features may be limited.
      </p>
      <Button className="mt-6" onClick={() => window.location.reload()}>
        Try Again
      </Button>
    </div>
  )
}
