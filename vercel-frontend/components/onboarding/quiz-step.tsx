"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Sparkles } from "lucide-react"

interface QuizStepProps {
  onComplete: (data: any) => void
}

const questions = [
  {
    id: "activity_level",
    question: "How would you describe your activity level?",
    type: "single",
    options: ["Low - Prefer relaxed activities", "Medium - Balanced mix", "High - Very active lifestyle"],
  },
  {
    id: "schedule",
    question: "When are you typically available for pet activities?",
    type: "multiple",
    options: ["Weekday mornings", "Weekday afternoons", "Weekday evenings", "Weekends"],
  },
  {
    id: "walk_frequency",
    question: "How often do you walk your pet?",
    type: "single",
    options: ["Once a day", "Twice a day", "Three or more times a day", "A few times a week"],
  },
  {
    id: "socialization",
    question: "How does your pet interact with other animals?",
    type: "single",
    options: [
      "Very social - loves all animals",
      "Selective - prefers certain types",
      "Cautious - needs slow introductions",
      "Prefers to be alone",
    ],
  },
  {
    id: "environment",
    question: "What's your living situation?",
    type: "single",
    options: ["House with yard", "Apartment with nearby parks", "Condo/Townhouse", "Rural/Farm setting"],
  },
  {
    id: "interests",
    question: "What activities interest you most?",
    type: "multiple",
    options: [
      "Dog park visits",
      "Hiking trails",
      "Beach outings",
      "Training classes",
      "Pet cafes",
      "Playdates at home",
    ],
  },
  {
    id: "experience",
    question: "What's your experience level with pets?",
    type: "single",
    options: ["First-time owner", "Some experience", "Very experienced", "Professional (trainer, vet, etc.)"],
  },
  {
    id: "group_size",
    question: "Preferred group size for activities?",
    type: "single",
    options: ["One-on-one playdates", "Small groups (2-3 pets)", "Medium groups (4-6 pets)", "Large groups (7+ pets)"],
  },
  {
    id: "distance",
    question: "How far are you willing to travel for meetups?",
    type: "single",
    options: ["Within 1 mile", "Within 3 miles", "Within 5 miles", "Within 10+ miles"],
  },
  {
    id: "commitment",
    question: "How often would you like to meet up?",
    type: "single",
    options: ["Daily", "A few times a week", "Weekly", "Bi-weekly", "Monthly", "Flexible/Spontaneous"],
  },
]

export function QuizStep({ onComplete }: QuizStepProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<Record<string, any>>({})

  const question = questions[currentQuestion]
  const isLastQuestion = currentQuestion === questions.length - 1
  const progress = ((currentQuestion + 1) / questions.length) * 100

  const handleAnswer = (value: string | string[]) => {
    setAnswers({ ...answers, [question.id]: value })
  }

  const handleNext = () => {
    if (isLastQuestion) {
      onComplete(answers)
    } else {
      setCurrentQuestion(currentQuestion + 1)
    }
  }

  const handleBack = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(currentQuestion - 1)
    }
  }

  const currentAnswer = answers[question.id]
  const isAnswered = question.type === "multiple" ? currentAnswer?.length > 0 : !!currentAnswer

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-primary">
          <Sparkles className="w-5 h-5" />
          <span className="text-sm font-medium">Compatibility Quiz</span>
        </div>
        <h1 className="text-3xl font-bold text-balance">Let's find your perfect matches</h1>
        <p className="text-muted-foreground">
          Question {currentQuestion + 1} of {questions.length}
        </p>
      </div>

      {/* Progress Bar */}
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <div className="h-full bg-primary transition-all duration-300" style={{ width: `${progress}%` }} />
      </div>

      {/* Question Card */}
      <Card className="glass p-6 space-y-6">
        <h2 className="text-xl font-semibold text-balance">{question.question}</h2>

        {question.type === "single" ? (
          <RadioGroup value={currentAnswer} onValueChange={handleAnswer}>
            <div className="space-y-3">
              {question.options.map((option) => (
                <Label
                  key={option}
                  htmlFor={option}
                  className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/50 cursor-pointer transition-colors"
                >
                  <RadioGroupItem value={option} id={option} />
                  <span className="flex-1">{option}</span>
                </Label>
              ))}
            </div>
          </RadioGroup>
        ) : (
          <div className="space-y-3">
            {question.options.map((option) => {
              const isChecked = currentAnswer?.includes(option)
              return (
                <Label
                  key={option}
                  htmlFor={option}
                  className="flex items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/50 cursor-pointer transition-colors"
                >
                  <Checkbox
                    id={option}
                    checked={isChecked}
                    onCheckedChange={(checked) => {
                      const current = currentAnswer || []
                      if (checked) {
                        handleAnswer([...current, option])
                      } else {
                        handleAnswer(current.filter((v: string) => v !== option))
                      }
                    }}
                  />
                  <span className="flex-1">{option}</span>
                </Label>
              )
            })}
          </div>
        )}
      </Card>

      {/* Navigation */}
      <div className="flex gap-3">
        {currentQuestion > 0 && (
          <Button type="button" variant="outline" onClick={handleBack} className="flex-1 bg-transparent">
            Back
          </Button>
        )}
        <Button type="button" onClick={handleNext} disabled={!isAnswered} className="flex-1">
          {isLastQuestion ? "Complete" : "Next"}
        </Button>
      </div>
    </div>
  )
}
