"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { OwnerInfoStep } from "@/components/onboarding/owner-info-step"
import { PetInfoStep } from "@/components/onboarding/pet-info-step"
import { QuizStep } from "@/components/onboarding/quiz-step"
import { Progress } from "@/components/ui/progress"
import { ChevronLeft } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function OnboardingPage() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)
  const [ownerData, setOwnerData] = useState<any>(null)
  const [petData, setPetData] = useState<any>(null)

  const totalSteps = 3
  const progress = (currentStep / totalSteps) * 100

  const handleOwnerComplete = (data: any) => {
    setOwnerData(data)
    setCurrentStep(2)
  }

  const handlePetComplete = (data: any) => {
    setPetData(data)
    setCurrentStep(3)
  }

  const handleQuizComplete = (data: any) => {
    // Save all onboarding data
    console.log("[v0] Onboarding complete:", { owner: ownerData, pet: petData, quiz: data })
    // Redirect to discover page
    router.push("/discover")
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    } else {
      router.push("/")
    }
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-40 glass-strong border-b border-border/50">
        <div className="flex items-center gap-4 px-4 py-4 max-w-lg mx-auto">
          <Button variant="ghost" size="icon" onClick={handleBack} className="shrink-0">
            <ChevronLeft className="w-5 h-5" />
          </Button>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">
                Step {currentStep} of {totalSteps}
              </span>
              <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="px-4 py-6 max-w-lg mx-auto">
        {currentStep === 1 && <OwnerInfoStep onComplete={handleOwnerComplete} initialData={ownerData} />}
        {currentStep === 2 && <PetInfoStep onComplete={handlePetComplete} initialData={petData} />}
        {currentStep === 3 && <QuizStep onComplete={handleQuizComplete} />}
      </main>
    </div>
  )
}
