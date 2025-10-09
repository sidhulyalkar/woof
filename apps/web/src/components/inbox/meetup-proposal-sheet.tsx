"use client"

import type React from "react"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetFooter } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Calendar, MapPin } from "lucide-react"

interface MeetupProposalSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSubmit: (data: { location: string; datetime: string; notes: string }) => void
}

export function MeetupProposalSheet({ open, onOpenChange, onSubmit }: MeetupProposalSheetProps) {
  const [formData, setFormData] = useState({
    location: "",
    datetime: "",
    notes: "",
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
    setFormData({ location: "", datetime: "", notes: "" })
  }

  const isValid = formData.location && formData.datetime

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[85vh] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Propose a Meetup</SheetTitle>
        </SheetHeader>

        <form onSubmit={handleSubmit} className="space-y-6 py-6">
          <div className="space-y-2">
            <Label htmlFor="location">Location *</Label>
            <div className="relative">
              <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="location"
                placeholder="e.g., Golden Gate Park, Dog Beach"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                className="pl-9"
                required
              />
            </div>
            <p className="text-xs text-muted-foreground">Enter a specific location or landmark</p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="datetime">Date & Time *</Label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                id="datetime"
                type="datetime-local"
                value={formData.datetime}
                onChange={(e) => setFormData({ ...formData, datetime: e.target.value })}
                className="pl-9"
                required
                min={new Date().toISOString().slice(0, 16)}
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="notes">Additional Notes (Optional)</Label>
            <Textarea
              id="notes"
              placeholder="Any special instructions or things to bring..."
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              rows={4}
              className="resize-none"
            />
          </div>

          <SheetFooter className="flex-row gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              className="flex-1 bg-transparent"
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!isValid} className="flex-1">
              Send Proposal
            </Button>
          </SheetFooter>
        </form>
      </SheetContent>
    </Sheet>
  )
}
