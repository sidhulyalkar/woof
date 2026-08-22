"use client"

import type React from "react"
import { useEffect, useState } from "react"
import { isAxiosError } from "axios"
import { Camera, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetFooter, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Textarea } from "@/components/ui/textarea"
import { storageApi } from "@/lib/api"
import { profileApi } from "@/lib/api/profile"
import type { AuthUser } from "@/lib/stores/auth-store"
import { useAuthStore } from "@/lib/stores/auth-store"

type ApiErrorBody = {
  message?: string | string[]
}

interface EditProfileSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  user: AuthUser
  onSaved?: (user: AuthUser) => void
}

export function EditProfileSheet({ open, onOpenChange, user, onSaved }: EditProfileSheetProps) {
  const [formData, setFormData] = useState({
    handle: user.handle,
    bio: user.bio || "",
    visibility: ("visibility" in user && typeof user.visibility === "string" ? user.visibility : "PUBLIC") as "PUBLIC" | "FRIENDS_ONLY" | "PRIVATE",
  })
  const [avatarFile, setAvatarFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState(user.avatarUrl || "")
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    setFormData({
      handle: user.handle,
      bio: user.bio || "",
      visibility: ("visibility" in user && typeof user.visibility === "string" ? user.visibility : "PUBLIC") as "PUBLIC" | "FRIENDS_ONLY" | "PRIVATE",
    })
    setAvatarFile(null)
    setPreviewUrl(user.avatarUrl || "")
  }, [open, user])

  const handleAvatar = (file?: File) => {
    if (!file) return
    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl)
    setAvatarFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    setIsSaving(true)

    try {
      let avatarUrl = user.avatarUrl || undefined
      if (avatarFile) {
        const upload = await storageApi.uploadFile(avatarFile, "users")
        avatarUrl = upload.url
      }

      const updated = await profileApi.update({
        handle: formData.handle.trim().toLowerCase().replace(/\s+/g, "_"),
        bio: formData.bio,
        visibility: formData.visibility,
        avatarUrl,
      })

      useAuthStore.getState().updateUser(updated)
      onSaved?.(updated)
      toast.success("Profile updated")
      onOpenChange(false)
    } catch (error: unknown) {
      const responseMessage = isAxiosError<ApiErrorBody>(error) ? error.response?.data?.message : undefined
      const message = responseMessage || "Profile could not be updated."
      toast.error(Array.isArray(message) ? message[0] : message)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[88vh] overflow-y-auto rounded-t-3xl border-border/70">
        <SheetHeader>
          <SheetTitle>Edit profile</SheetTitle>
        </SheetHeader>

        <form onSubmit={handleSubmit} className="mx-auto max-w-lg space-y-6 py-6">
          <div className="flex flex-col items-center gap-3">
            <div className="relative">
              <Avatar className="h-24 w-24 border-2 border-border">
                <AvatarImage src={previewUrl || "/placeholder.svg"} alt="Profile preview" />
                <AvatarFallback>{user.handle.slice(0, 1).toUpperCase()}</AvatarFallback>
              </Avatar>
              <Label htmlFor="profile-avatar" className="absolute bottom-0 right-0 flex h-9 w-9 cursor-pointer items-center justify-center rounded-full border border-border bg-background text-foreground shadow-lg">
                <Camera className="h-4 w-4" aria-hidden="true" />
                <span className="sr-only">Choose profile photo</span>
              </Label>
              <Input id="profile-avatar" type="file" accept="image/jpeg,image/png,image/webp" className="hidden" onChange={(event) => handleAvatar(event.target.files?.[0])} />
            </div>
            <p className="text-xs text-muted-foreground">Optional JPEG, PNG, or WebP.</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="profile-handle">Public handle</Label>
            <Input
              id="profile-handle"
              value={formData.handle}
              onChange={(event) => setFormData({ ...formData, handle: event.target.value })}
              minLength={3}
              maxLength={30}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="profile-bio">Bio</Label>
            <Textarea
              id="profile-bio"
              placeholder="What kinds of dog-friendly activities are you usually up for?"
              value={formData.bio}
              onChange={(event) => setFormData({ ...formData, bio: event.target.value.slice(0, 500) })}
              rows={4}
              className="resize-none"
            />
            <p className="text-right text-xs text-muted-foreground">{formData.bio.length}/500</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="profile-visibility">Profile visibility</Label>
            <Select value={formData.visibility} onValueChange={(visibility: "PUBLIC" | "FRIENDS_ONLY" | "PRIVATE") => setFormData({ ...formData, visibility })}>
              <SelectTrigger id="profile-visibility">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="PUBLIC">Public</SelectItem>
                <SelectItem value="FRIENDS_ONLY">Friends only</SelectItem>
                <SelectItem value="PRIVATE">Private</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs leading-relaxed text-muted-foreground">Visibility controls profile discovery. Precise live or home location requires separate contextual permission.</p>
          </div>

          <SheetFooter className="flex-row gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={isSaving} className="flex-1 bg-transparent">
              Cancel
            </Button>
            <Button type="submit" disabled={isSaving} className="flex-1">
              {isSaving ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                  Saving…
                </>
              ) : (
                "Save changes"
              )}
            </Button>
          </SheetFooter>
        </form>
      </SheetContent>
    </Sheet>
  )
}
