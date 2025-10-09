"use client"

import type React from "react"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { CheckCircle2, Star } from "lucide-react"
import type { Event } from "@/lib/types"
import { cn } from "@/lib/utils"

interface CheckInSheetProps {
  event: Event
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function CheckInSheet({ event, open, onOpenChange }: CheckInSheetProps) {
  const [checkedIn, setCheckedIn] = useState(false)
  const [rating, setRating] = useState<string>("")
  const [feedback, setFeedback] = useState("")

  const handleCheckIn = () => {
    setCheckedIn(true)
  }

  const handleSubmitFeedback = (e: React.FormEvent) => {
    e.preventDefault()
    console.log("[v0] Event feedback:", { eventId: event.id, rating, feedback })
    onOpenChange(false)
    // Reset form
    setCheckedIn(false)
    setRating("")
    setFeedback("")
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[85vh] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{checkedIn ? "Rate Your Experience" : "Check In"}</SheetTitle>
        </SheetHeader>

        {!checkedIn ? (
          <div className="space-y-6 py-6">
            <div className="text-center space-y-4">
              <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-10 h-10 text-primary" />
              </div>
              <div>
                <h3 className="text-xl font-semibold mb-2">{event.title}</h3>
                <p className="text-sm text-muted-foreground">{event.location.address}</p>
              </div>
            </div>

            <div className="glass rounded-lg p-6 space-y-3 text-center">
              <p className="text-sm text-muted-foreground">
                Check in to confirm your attendance and unlock the ability to rate this event after it ends.
              </p>
            </div>

            <Button size="lg" className="w-full gap-2" onClick={handleCheckIn}>
              <CheckCircle2 className="w-5 h-5" />
              Check In Now
            </Button>
          </div>
        ) : (
          <form onSubmit={handleSubmitFeedback} className="space-y-6 py-6">
            <div className="text-center space-y-2">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Checked In Successfully!</h3>
              <p className="text-sm text-muted-foreground">How was your experience?</p>
            </div>

            <div className="space-y-3">
              <Label>Rate this event</Label>
              <RadioGroup value={rating} onValueChange={setRating}>
                <div className="space-y-2">
                  {[
                    { value: "5", label: "Excellent", emoji: "🌟" },
                    { value: "4", label: "Great", emoji: "😊" },
                    { value: "3", label: "Good", emoji: "🙂" },
                    { value: "2", label: "Fair", emoji: "😐" },
                    { value: "1", label: "Poor", emoji: "😞" },
                  ].map((option) => (
                    <Label
                      key={option.value}
                      htmlFor={option.value}
                      className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/50 cursor-pointer transition-colors"
                    >
                      <RadioGroupItem value={option.value} id={option.value} />
                      <span className="text-2xl">{option.emoji}</span>
                      <div className="flex-1">
                        <span className="font-medium">{option.label}</span>
                        <div className="flex gap-0.5 mt-1">
                          {Array.from({ length: 5 }).map((_, i) => (
                            <Star
                              key={i}
                              className={cn(
                                "w-3 h-3",
                                i < Number.parseInt(option.value) ? "fill-primary text-primary" : "text-muted",
                              )}
                            />
                          ))}
                        </div>
                      </div>
                    </Label>
                  ))}
                </div>
              </RadioGroup>
            </div>

            <div className="space-y-2">
              <Label htmlFor="feedback">Additional Feedback (Optional)</Label>
              <Textarea
                id="feedback"
                placeholder="Share your thoughts about the event..."
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                rows={4}
                className="resize-none"
              />
            </div>

            <div className="flex gap-3">
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                className="flex-1 bg-transparent"
              >
                Skip
              </Button>
              <Button type="submit" disabled={!rating} className="flex-1">
                Submit Feedback
              </Button>
            </div>
          </form>
        )}
      </SheetContent>
    </Sheet>
  )
}
