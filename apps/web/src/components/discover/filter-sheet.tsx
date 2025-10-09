"use client"

import { useState } from "react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetFooter } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"

interface FilterSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const speciesOptions = ["Dog", "Cat", "Other"]
const sizeOptions = ["Small", "Medium", "Large"]
const activityLevels = ["Low", "Medium", "High"]

export function FilterSheet({ open, onOpenChange }: FilterSheetProps) {
  const [distance, setDistance] = useState([10])
  const [minCompatibility, setMinCompatibility] = useState([70])
  const [species, setSpecies] = useState<string[]>(["Dog"])
  const [sizes, setSizes] = useState<string[]>([])
  const [activityLevel, setActivityLevel] = useState<string[]>([])

  const toggleOption = (value: string, current: string[], setter: (val: string[]) => void) => {
    if (current.includes(value)) {
      setter(current.filter((v) => v !== value))
    } else {
      setter([...current, value])
    }
  }

  const handleReset = () => {
    setDistance([10])
    setMinCompatibility([70])
    setSpecies(["Dog"])
    setSizes([])
    setActivityLevel([])
  }

  const handleApply = () => {
    // Apply filters logic here
    onOpenChange(false)
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="bottom" className="h-[85vh] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Filter Matches</SheetTitle>
        </SheetHeader>

        <div className="space-y-6 py-6">
          {/* Distance */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Maximum Distance</Label>
              <span className="text-sm font-medium text-primary">{distance[0]} miles</span>
            </div>
            <Slider value={distance} onValueChange={setDistance} min={1} max={50} step={1} />
          </div>

          {/* Compatibility */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label>Minimum Compatibility</Label>
              <span className="text-sm font-medium text-primary">{minCompatibility[0]}%</span>
            </div>
            <Slider value={minCompatibility} onValueChange={setMinCompatibility} min={50} max={100} step={5} />
          </div>

          {/* Species */}
          <div className="space-y-3">
            <Label>Pet Species</Label>
            <div className="flex flex-wrap gap-2">
              {speciesOptions.map((option) => {
                const isSelected = species.includes(option)
                return (
                  <Badge
                    key={option}
                    variant={isSelected ? "default" : "outline"}
                    className="cursor-pointer hover:bg-primary/20 transition-colors"
                    onClick={() => toggleOption(option, species, setSpecies)}
                  >
                    {option}
                  </Badge>
                )
              })}
            </div>
          </div>

          {/* Size */}
          <div className="space-y-3">
            <Label>Pet Size</Label>
            <div className="flex flex-wrap gap-2">
              {sizeOptions.map((option) => {
                const isSelected = sizes.includes(option)
                return (
                  <Badge
                    key={option}
                    variant={isSelected ? "default" : "outline"}
                    className="cursor-pointer hover:bg-primary/20 transition-colors"
                    onClick={() => toggleOption(option, sizes, setSizes)}
                  >
                    {option}
                  </Badge>
                )
              })}
            </div>
          </div>

          {/* Activity Level */}
          <div className="space-y-3">
            <Label>Activity Level</Label>
            <div className="flex flex-wrap gap-2">
              {activityLevels.map((option) => {
                const isSelected = activityLevel.includes(option)
                return (
                  <Badge
                    key={option}
                    variant={isSelected ? "default" : "outline"}
                    className="cursor-pointer hover:bg-primary/20 transition-colors"
                    onClick={() => toggleOption(option, activityLevel, setActivityLevel)}
                  >
                    {option}
                  </Badge>
                )
              })}
            </div>
          </div>

          {/* Availability */}
          <div className="space-y-3">
            <Label>Availability</Label>
            <div className="space-y-2">
              {["Weekday mornings", "Weekday afternoons", "Weekday evenings", "Weekends"].map((time) => (
                <Label key={time} className="flex items-center gap-2 cursor-pointer">
                  <Checkbox />
                  <span className="text-sm">{time}</span>
                </Label>
              ))}
            </div>
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
