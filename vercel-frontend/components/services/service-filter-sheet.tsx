"use client"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetFooter } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"

interface ServiceFilterSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const serviceTypes = ["Dog Walking", "Pet Sitting", "Training", "Grooming"]
const availabilityOptions = ["Weekday mornings", "Weekday afternoons", "Weekday evenings", "Weekends"]

export function ServiceFilterSheet({ open, onOpenChange }: ServiceFilterSheetProps) {
  const [distance, setDistance] = useState([5])
  const [minRating, setMinRating] = useState([4])
  const [selectedServices, setSelectedServices] = useState<string[]>([])
  const [verifiedOnly, setVerifiedOnly] = useState(false)

  const toggleService = (service: string) => {
    if (selectedServices.includes(service)) {
      setSelectedServices(selectedServices.filter((s) => s !== service))
    } else {
      setSelectedServices([...selectedServices, service])
    }
  }

  const handleReset = () => {
    setDistance([5])
    setMinRating([4])
    setSelectedServices([])
    setVerifiedOnly(false)
  }

  const handleApply = () => {
    onOpenChange(false)
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[85vh] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Filter Services</SheetTitle>
        </SheetHeader>

        <div className="space-y-6 py-6">
          {/* Distance */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Maximum Distance</Label>
              <span className="text-sm font-medium text-primary">{distance[0]} miles</span>
            </div>
            <Slider value={distance} onValueChange={setDistance} min={1} max={25} step={1} />
          </div>

          {/* Rating */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Minimum Rating</Label>
              <span className="text-sm font-medium text-primary">{minRating[0]} stars</span>
            </div>
            <Slider value={minRating} onValueChange={setMinRating} min={1} max={5} step={0.5} />
          </div>

          {/* Service Types */}
          <div className="space-y-3">
            <Label>Service Types</Label>
            <div className="flex flex-wrap gap-2">
              {serviceTypes.map((service) => {
                const isSelected = selectedServices.includes(service)
                return (
                  <Badge
                    key={service}
                    variant={isSelected ? "default" : "outline"}
                    className="cursor-pointer hover:bg-primary/20 transition-colors"
                    onClick={() => toggleService(service)}
                  >
                    {service}
                  </Badge>
                )
              })}
            </div>
          </div>

          {/* Availability */}
          <div className="space-y-3">
            <Label>Availability</Label>
            <div className="space-y-2">
              {availabilityOptions.map((option) => (
                <Label key={option} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox />
                  <span className="text-sm">{option}</span>
                </Label>
              ))}
            </div>
          </div>

          {/* Verified Only */}
          <div className="space-y-3">
            <Label className="flex items-center gap-2 cursor-pointer">
              <Checkbox checked={verifiedOnly} onCheckedChange={(checked) => setVerifiedOnly(checked as boolean)} />
              <span>Show verified providers only</span>
            </Label>
          </div>
        </div>

        <SheetFooter className="flex-row gap-2">
          <Button variant="outline" onClick={handleReset} className="flex-1 bg-transparent">
            Reset
          </Button>
          <Button onClick={handleApply} className="flex-1">
            Apply Filters
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
