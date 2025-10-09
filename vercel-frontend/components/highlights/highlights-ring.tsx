"use client"

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import Link from "next/link"

interface HighlightsRingProps {
  userId: string
  userName: string
  userAvatar: string
  petAvatar: string
  hasUnviewed?: boolean
}

export function HighlightsRing({ userId, userName, userAvatar, petAvatar, hasUnviewed = false }: HighlightsRingProps) {
  return (
    <Link href="/highlights" className="flex flex-col items-center gap-1">
      <div
        className={`relative rounded-full p-0.5 ${
          hasUnviewed ? "bg-gradient-to-br from-accent to-secondary" : "bg-border"
        }`}
      >
        <div className="rounded-full bg-background p-0.5">
          <div className="relative">
            <Avatar className="h-16 w-16 border-2 border-background">
              <AvatarImage src={userAvatar || "/placeholder.svg"} alt={userName} />
              <AvatarFallback>{userName[0]}</AvatarFallback>
            </Avatar>
            <Avatar className="absolute -bottom-1 -right-1 h-6 w-6 border-2 border-background">
              <AvatarImage src={petAvatar || "/placeholder.svg"} alt="Pet" />
              <AvatarFallback>P</AvatarFallback>
            </Avatar>
          </div>
        </div>
      </div>
      <p className="text-xs text-muted-foreground truncate max-w-[70px]">{userName.split(" ")[0]}</p>
    </Link>
  )
}
