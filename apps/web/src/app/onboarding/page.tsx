"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { OwnerInfoStep } from "@/components/onboarding/owner-info-step"
import { PetInfoStep } from "@/components/onboarding/pet-info-step"
import { QuizStep } from "@/components/onboarding/quiz-step"
import { Progress } from "@/components/ui/progress"
import { ChevronLeft, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { authApi, petsApi } from "@/lib/api"
import { useSessionStore } from "@/store/session"
import { toast } from "sonner"

export default function OnboardingPage() {
  const router = useRouter()
  const setSession = useSessionStore(state => state.setSession)
  const [currentStep, setCurrentStep] = useState(1)
  const [ownerData, setOwnerData] = useState<any>(null)
  const [petData, setPetData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

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

  const handleQuizComplete = async (data: any) => {
    setIsLoading(true)
    setError('')

    try {
      // Step 1: Register user
      const registerResult = await authApi.register({
        handle: ownerData.name.toLowerCase().replace(/\s+/g, '_'),
        email: ownerData.email,
        password: ownerData.password,
        bio: ownerData.bio || undefined,
      })

      const { user, token } = registerResult

      // Step 2: Save session
      setSession(user, token)
      localStorage.setItem('authToken', token)

      // Step 3: Create pet
      await petsApi.createPet({
        name: petData.name,
        species: petData.species,
        breed: petData.breed,
        age: parseInt(petData.age),
        size: petData.size,
        temperament: petData.temperament,
        medicalNotes: petData.medicalNotes || undefined,
      })

      toast.success('Welcome to PetPath! 🎉')

      // Redirect to home
      router.push('/')
    } catch (err: any) {
      console.error('Onboarding failed:', err)
      const errorMessage = err?.response?.data?.message || 'Registration failed. Please try again.'
      setError(errorMessage)
      toast.error(errorMessage)
    } finally {
      setIsLoading(false)
    }
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
        {currentStep === 3 && <QuizStep onComplete={handleQuizComplete} isLoading={isLoading} />}

        {error && currentStep === 3 && (
          <div className="mt-4 text-sm text-destructive bg-destructive/10 p-3 rounded-md">
            {error}
          </div>
        )}
      </main>
    </div>
  )
}
