"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Upload, PawPrint, X } from "lucide-react"

interface PetInfoStepProps {
  onComplete: (data: any) => void
  initialData?: any
}

const temperamentOptions = ["Friendly", "Energetic", "Calm", "Playful", "Shy", "Protective", "Social", "Independent"]

export function PetInfoStep({ onComplete, initialData }: PetInfoStepProps) {
  const [formData, setFormData] = useState({
    name: initialData?.name || "",
    species: initialData?.species || "",
    breed: initialData?.breed || "",
    age: initialData?.age || "",
    size: initialData?.size || "",
    temperament: initialData?.temperament || [],
    photoUrl: initialData?.photoUrl || "",
    medicalNotes: initialData?.medicalNotes || "",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onComplete(formData)
  }

  const toggleTemperament = (trait: string) => {
    const current = formData.temperament
    if (current.includes(trait)) {
      setFormData({ ...formData, temperament: current.filter((t: string) => t !== trait) })
    } else {
      setFormData({ ...formData, temperament: [...current, trait] })
    }
  }

  const isValid = formData.name && formData.species && formData.breed && formData.age && formData.size

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-balance">Tell us about your pet</h1>
        <p className="text-muted-foreground text-pretty">This helps us find compatible playmates and activities.</p>
      </div>

      {/* Pet Photo Upload */}
      <Card className="glass p-6">
        <div className="flex items-center gap-4">
          <Avatar className="w-20 h-20">
            <AvatarImage src={formData.photoUrl || "/placeholder.svg"} />
            <AvatarFallback className="bg-secondary/10">
              <PawPrint className="w-8 h-8 text-secondary" />
            </AvatarFallback>
          </Avatar>
          <div className="flex-1">
            <Label htmlFor="pet-photo" className="cursor-pointer">
              <div className="flex items-center gap-2 text-sm text-primary hover:text-primary/80 transition-colors">
                <Upload className="w-4 h-4" />
                <span>Upload Pet Photo</span>
              </div>
            </Label>
            <Input
              id="pet-photo"
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0]
                if (file) {
                  const url = URL.createObjectURL(file)
                  setFormData({ ...formData, photoUrl: url })
                }
              }}
            />
            <p className="text-xs text-muted-foreground mt-1">Show off your furry friend!</p>
          </div>
        </div>
      </Card>

      {/* Form Fields */}
      <div className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="pet-name">Pet Name *</Label>
          <Input
            id="pet-name"
            placeholder="Enter your pet's name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="species">Species *</Label>
            <Select value={formData.species} onValueChange={(value) => setFormData({ ...formData, species: value })}>
              <SelectTrigger id="species">
                <SelectValue placeholder="Select" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dog">Dog</SelectItem>
                <SelectItem value="cat">Cat</SelectItem>
                <SelectItem value="other">Other</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="pet-age">Age *</Label>
            <Input
              id="pet-age"
              type="number"
              placeholder="Years"
              value={formData.age}
              onChange={(e) => setFormData({ ...formData, age: e.target.value })}
              required
              min="0"
              max="30"
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="breed">Breed *</Label>
          <Input
            id="breed"
            placeholder="e.g., Golden Retriever, Tabby"
            value={formData.breed}
            onChange={(e) => setFormData({ ...formData, breed: e.target.value })}
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="size">Size *</Label>
          <Select value={formData.size} onValueChange={(value) => setFormData({ ...formData, size: value })}>
            <SelectTrigger id="size">
              <SelectValue placeholder="Select size" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="small">Small (0-25 lbs)</SelectItem>
              <SelectItem value="medium">Medium (26-60 lbs)</SelectItem>
              <SelectItem value="large">Large (60+ lbs)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Temperament</Label>
          <p className="text-xs text-muted-foreground mb-3">Select all that apply</p>
          <div className="flex flex-wrap gap-2">
            {temperamentOptions.map((trait) => {
              const isSelected = formData.temperament.includes(trait)
              return (
                <Badge
                  key={trait}
                  variant={isSelected ? "default" : "outline"}
                  className="cursor-pointer hover:bg-primary/20 transition-colors"
                  onClick={() => toggleTemperament(trait)}
                >
                  {trait}
                  {isSelected && <X className="w-3 h-3 ml-1" />}
                </Badge>
              )
            })}
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="medical">Medical Notes (Optional)</Label>
          <Textarea
            id="medical"
            placeholder="Any medical conditions, allergies, or special needs..."
            value={formData.medicalNotes}
            onChange={(e) => setFormData({ ...formData, medicalNotes: e.target.value })}
            rows={3}
            className="resize-none"
          />
        </div>
      </div>

      <Button type="submit" size="lg" className="w-full" disabled={!isValid}>
        Continue
      </Button>
    </form>
  )
}
