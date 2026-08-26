"use client"

import { useState } from "react"
import { Clock3, HeartHandshake, ShieldCheck, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import {
  emptyFirstAdventureSelections,
  type FirstAdventureEffort,
  type FirstAdventureGoal,
  type FirstAdventureSelections,
  type FirstAdventureSocialComfort,
  type FirstAdventureTimeBudget,
} from "@/lib/onboarding/first-adventure"

interface FirstAdventureStepProps {
  petName: string
  isLoading?: boolean
  onComplete: (selections: FirstAdventureSelections) => void
  onSkipAll: () => void
}

const goalOptions: Array<{ value: FirstAdventureGoal; label: string; detail: string }> = [
  { value: "MORE_ADVENTURES", label: "More little adventures", detail: "Explore and move together" },
  { value: "TRAINING", label: "Train together", detail: "Build useful skills as a team" },
  { value: "CALMER_ROUTINES", label: "Calmer routines", detail: "Make everyday moments easier" },
  { value: "SOCIAL_CONFIDENCE", label: "Social confidence", detail: "Respectful comfort around others" },
  { value: "CARE_ROUTINES", label: "Care routines", detail: "Handling and everyday care" },
  { value: "JUST_HAVE_FUN", label: "Mostly have fun", detail: "More good moments together" },
]

const timeOptions: Array<{ value: FirstAdventureTimeBudget; label: string }> = [
  { value: "FIVE_MIN", label: "About 5 min" },
  { value: "TEN_TO_FIFTEEN", label: "10–15 min" },
  { value: "TWENTY_TO_THIRTY", label: "20–30 min" },
  { value: "FORTY_PLUS", label: "40+ min" },
  { value: "VARIES", label: "It varies" },
]

const effortOptions: Array<{ value: FirstAdventureEffort; label: string }> = [
  { value: "KEEP_IT_EASY", label: "Keep it easy" },
  { value: "MODERATE", label: "A little effort" },
  { value: "UP_FOR_A_CHALLENGE", label: "Up for a challenge" },
  { value: "VARIES", label: "Depends on the day" },
]

const socialOptions: Array<{ value: FirstAdventureSocialComfort; label: string; detail: string }> = [
  { value: "PREFERS_SPACE", label: "Usually prefers space", detail: "Distance is part of the plan" },
  { value: "CALM_AT_DISTANCE", label: "Comfortable at a distance", detail: "Nearby without needing contact" },
  { value: "SELECTIVELY_SOCIAL", label: "Selective about friends", detail: "Some dogs are a good fit" },
  { value: "OFTEN_SOCIAL", label: "Often enjoys other dogs", detail: "Still with choice and space" },
  { value: "NOT_SURE", label: "Not sure yet", detail: "Woof can learn gently over time" },
]

function SelectChip({
  selected,
  disabled,
  onClick,
  children,
}: {
  selected: boolean
  disabled?: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "min-h-11 rounded-xl border px-3 py-2 text-left text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-60",
        selected
          ? "border-primary/50 bg-primary/12 text-foreground"
          : "border-border bg-background/65 text-muted-foreground hover:border-primary/30 hover:text-foreground"
      )}
    >
      {children}
    </button>
  )
}

export function FirstAdventureStep({
  petName,
  isLoading = false,
  onComplete,
  onSkipAll,
}: FirstAdventureStepProps) {
  const [selections, setSelections] = useState<FirstAdventureSelections>(
    emptyFirstAdventureSelections()
  )

  const toggleGoal = (goal: FirstAdventureGoal) => {
    setSelections((current) => {
      if (current.goals.includes(goal)) {
        return { ...current, goals: current.goals.filter((value) => value !== goal) }
      }
      if (current.goals.length >= 3) return current
      return { ...current, goals: [...current.goals, goal] }
    })
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="eyebrow">First Adventure</p>
        <h1 className="text-3xl font-bold tracking-tight text-balance">
          Let&apos;s make the first suggestion feel like the two of you
        </h1>
        <p className="text-sm leading-relaxed text-muted-foreground text-pretty">
          Three small moments are enough to start. Skip anything you do not know yet. Woof can learn from what actually works for you and {petName} over time.
        </p>
      </div>

      <Card className="glass space-y-4 rounded-2xl p-5">
        <div className="flex gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
            <Sparkles className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h2 className="font-semibold">1. What would feel useful together right now?</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Pick up to three. This helps choose between equally safe Adventures, not grade your goals.
            </p>
          </div>
        </div>

        <div className="grid gap-2 sm:grid-cols-2">
          {goalOptions.map((option) => {
            const selected = selections.goals.includes(option.value)
            const disabled = !selected && selections.goals.length >= 3
            return (
              <SelectChip
                key={option.value}
                selected={selected}
                disabled={isLoading || disabled}
                onClick={() => toggleGoal(option.value)}
              >
                <span className="block">{option.label}</span>
                <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                  {option.detail}
                </span>
              </SelectChip>
            )
          })}
        </div>

        <details className="rounded-xl border border-border/70 bg-background/50 px-3 py-2 text-xs text-muted-foreground">
          <summary className="cursor-pointer font-semibold text-foreground">Why are you asking?</summary>
          <p className="mt-2 leading-relaxed">
            If two activities are both safe, your current goals can break the tie. You can change them later, and real outcomes can refine what Woof suggests.
          </p>
        </details>
      </Card>

      <Card className="glass space-y-4 rounded-2xl p-5">
        <div className="flex gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
            <Clock3 className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h2 className="font-semibold">2. What tends to fit real life?</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              A five-minute win is better than a perfect plan that never fits the day.
            </p>
          </div>
        </div>

        <fieldset className="space-y-2">
          <legend className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Comfortable session length
          </legend>
          <div className="flex flex-wrap gap-2">
            {timeOptions.map((option) => (
              <SelectChip
                key={option.value}
                selected={selections.timeBudget === option.value}
                disabled={isLoading}
                onClick={() =>
                  setSelections((current) => ({
                    ...current,
                    timeBudget: current.timeBudget === option.value ? null : option.value,
                  }))
                }
              >
                {option.label}
              </SelectChip>
            ))}
          </div>
        </fieldset>

        <fieldset className="space-y-2">
          <legend className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Effort that usually feels good
          </legend>
          <div className="flex flex-wrap gap-2">
            {effortOptions.map((option) => (
              <SelectChip
                key={option.value}
                selected={selections.effort === option.value}
                disabled={isLoading}
                onClick={() =>
                  setSelections((current) => ({
                    ...current,
                    effort: current.effort === option.value ? null : option.value,
                  }))
                }
              >
                {option.label}
              </SelectChip>
            ))}
          </div>
        </fieldset>
      </Card>

      <Card className="glass space-y-4 rounded-2xl p-5">
        <div className="flex gap-3">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent/20 text-foreground">
            <ShieldCheck className="h-5 w-5" aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <h2 className="font-semibold">3. What should Woof respect from the start?</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              For now, one high-value comfort signal is enough. Unknown is a perfectly useful answer.
            </p>
          </div>
        </div>

        <div className="grid gap-2">
          {socialOptions.map((option) => (
            <SelectChip
              key={option.value}
              selected={selections.socialComfort === option.value}
              disabled={isLoading}
              onClick={() =>
                setSelections((current) => ({
                  ...current,
                  socialComfort: current.socialComfort === option.value ? null : option.value,
                }))
              }
            >
              <span className="block">{option.label}</span>
              <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                {option.detail}
              </span>
            </SelectChip>
          ))}
        </div>

        <details className="rounded-xl border border-border/70 bg-background/50 px-3 py-2 text-xs text-muted-foreground">
          <summary className="cursor-pointer font-semibold text-foreground">Why are you asking?</summary>
          <p className="mt-2 leading-relaxed">
            Social Adventures should start from known comfort and choice. Woof will not assume that more interaction is always better, and one unusual day will not rewrite this on its own.
          </p>
        </details>
      </Card>

      <Card className="surface-soft flex gap-3 rounded-2xl p-4">
        <HeartHandshake className="mt-0.5 h-5 w-5 shrink-0 text-primary" aria-hidden="true" />
        <p className="text-xs leading-relaxed text-muted-foreground">
          There is no profile score to finish. After Adventures, Woof can ask one tiny question only when the answer would meaningfully improve the next suggestion. Skipping never costs progress.
        </p>
      </Card>

      <div className="space-y-2">
        <Button
          type="button"
          size="lg"
          className="w-full"
          disabled={isLoading}
          onClick={() => onComplete(selections)}
        >
          {isLoading ? "Finding a good starting point…" : "Find our first Adventure"}
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="lg"
          className="w-full text-muted-foreground"
          disabled={isLoading}
          onClick={onSkipAll}
        >
          Skip personalization for now
        </Button>
      </div>
    </div>
  )
}
