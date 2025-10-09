"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Upload, User } from "lucide-react"

interface OwnerInfoStepProps {
  onComplete: (data: any) => void
  initialData?: any
}

export function OwnerInfoStep({ onComplete, initialData }: OwnerInfoStepProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    age: initialData?.age || "",
    location: initialData?.location || "",
    bio: initialData?.bio || "",
    avatarUrl: initialData?.avatarUrl || "",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onComplete(formData)
  }

  const isValid = formData.name && formData.age && formData.location

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-balance">Tell us about yourself</h1>
        <p className="text-muted-foreground text-pretty">Help us find the perfect matches for you and your pet.</p>
      </div>

      {/* Avatar Upload */}
      <Card className="glass p-6">
        <div className="flex items-center gap-4">
          <Avatar className="w-20 h-20">
            <AvatarImage src={formData.avatarUrl || "/placeholder.svg"} />
            <AvatarFallback className="bg-primary/10">
              <User className="w-8 h-8 text-primary" />
            </AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <Label htmlFor="avatar" className="cursor-pointer">
              <div className="flex items-center gap-2 text-sm text-primary hover:text-primary/80 transition-colors">
                <Upload className="w-4 h-4" />
                <span>Upload Photo</span>
              </div>
            </Label>
            <Input
              id="avatar"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) {
                  const url = URL.createObjectURL(file)
                  setFormData({ ...formData, avatarUrl: url })
                }
              }}
            />
            <p className="text-xs text-muted-foreground mt-1">JPG, PNG or GIF (max 5MB)</p>
          </div>
        </div>
      </Card>

      {/* Form Fields */}
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="name">Full Name *</Label>
          <Input
            id="name"
            placeholder="Enter your name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="age">Age *</Label>
          <Input
            id="age"
            type="number"
            placeholder="Enter your age"
            value={formData.age}
            onChange={(e) => setFormData({ ...formData, age: e.target.value })}
            required
            min="18"
            max="120"
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="location">Location *</Label>
          <Input
            id="location"
            placeholder="City, State"
            value={formData.location}
            onChange={(e) => setFormData({ ...formData, location: e.target.value })}
            required
          />
          <p className="text-xs text-muted-foreground">We'll use this to find nearby pet owners</p>
        </div>

        <div className="space-y-2">
          <Label htmlFor="bio">Bio (Optional)</Label>
          <Textarea
            id="bio"
            placeholder="Tell us a bit about yourself and what you're looking for..."
            value={formData.bio}
            onChange={(e) => setFormData({ ...formData, bio: e.target.value })}
            rows={4}
            className="resize-none"
          />
          <p className="text-xs text-muted-foreground">{formData.bio.length}/500 characters</p>
        </div>
      </div>

      <Button type="submit" size="lg" className="w-full" disabled={!isValid}>
        Continue
      </Button>
    </form>
  )
}
