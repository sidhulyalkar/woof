"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { isAxiosError } from "axios"
import { ChevronLeft, PawPrint } from "lucide-react"
import { toast } from "sonner"
import { OwnerInfoStep, type OwnerInfoData } from "@/components/onboarding/owner-info-step"
import { PetInfoStep, type PetInfoData } from "@/components/onboarding/pet-info-step"
import { QuizStep } from "@/components/onboarding/quiz-step"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { authApi, petsApi, storageApi } from "@/lib/api"
import { quizApi, type QuizAnswers } from "@/lib/api/quiz"
import { useAuthStore } from "@/lib/stores/auth-store"

type ApiErrorBody = {
  message?: string | string[]
}

export default function OnboardingPage() {
  const router = useRouter()
  const [currentStep, setCurrentStep] = useState(1)
  const [ownerData, setOwnerData] = useState<OwnerInfoData | null>(null)
  const [petData, setPetData] = useState<PetInfoData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")

  const totalSteps = 3
  const progress = (currentStep / totalSteps) * 100

  const handleOwnerComplete = (data: OwnerInfoData) => {
    setOwnerData(data)
    setCurrentStep(2)
  }

  const handlePetComplete = (data: PetInfoData) => {
    setPetData(data)
    setCurrentStep(3)
  }

  const handleQuizComplete = async (answers: QuizAnswers) => {
    if (!ownerData || !petData) {
      setError("Your onboarding details are incomplete. Go back and review the earlier steps.")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      const authState = useAuthStore.getState()

      // Registration may already have succeeded during an earlier retry.
      if (!authState.isAuthenticated || !authState.token) {
        await authApi.register({
          handle: ownerData.handle,
          email: ownerData.email,
          password: ownerData.password,
          bio: ownerData.bio || undefined,
        })
      }

      let avatarUrl: string | undefined
      if (petData.photoFile) {
        try {
          const upload = await storageApi.uploadFile(petData.photoFile, "pets")
          avatarUrl = upload.url
        } catch (uploadError) {
          console.warn("Pet photo upload failed during onboarding", uploadError)
          toast.warning("Your account is safe, but the pet photo could not be uploaded. You can add it later.")
        }
      }

      const pet = await petsApi.createPet({
        name: petData.name,
        species: petData.species,
        breed: petData.breed,
        birthdate: petData.birthdate,
        temperament: petData.temperament,
        avatarUrl,
      })

      try {
        await quizApi.saveResponses({
          sessionId: `onboarding-${Date.now()}`,
          petId: pet.id,
          responses: answers,
        })
      } catch (quizError) {
        console.warn("Preference persistence failed during onboarding", quizError)
        toast.warning("Your profile was created, but matching preferences could not be saved. You can retake them later.")
      }

      // Refresh the canonical persisted auth profile so pet-aware screens are immediately correct.
      const profile = await authApi.me()
      const token = useAuthStore.getState().token
      if (token) {
        useAuthStore.getState().setAuth(profile, token)
      }

      toast.success("Welcome to Woof 🐾")
      router.replace("/")
    } catch (err: unknown) {
      console.error("Onboarding failed", err)
      const responseMessage = isAxiosError<ApiErrorBody>(err) ? err.response?.data?.message : undefined
      const message =
        responseMessage ||
        "We could not finish the profile. Your completed account step is preserved, so you can safely try again."
      setError(Array.isArray(message) ? message.join(" ") : message)
      toast.error(Array.isArray(message) ? message[0] : message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleBack = () => {
    if (isLoading) return
    if (currentStep > 1) {
      setCurrentStep((step) => step - 1)
    } else {
      router.push("/login")
    }
  }

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/88 backdrop-blur-2xl">
        <div className="mx-auto flex max-w-lg items-center gap-4 px-4 py-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={handleBack}
            disabled={isLoading}
            className="shrink-0 rounded-xl"
            aria-label={currentStep > 1 ? "Go to previous onboarding step" : "Return to sign in"}
          >
            <ChevronLeft className="h-5 w-5" aria-hidden="true" />
          </Button>
          <div className="flex-1">
            <div className="mb-2 flex items-center justify-between gap-4">
              <span className="text-sm font-semibold">Step {currentStep} of {totalSteps}</span>
              <span className="text-xs text-muted-foreground">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" aria-label={`Onboarding ${Math.round(progress)}% complete`} />
          </div>
          <span className="brand-mark flex h-9 w-9 shrink-0 items-center justify-center rounded-xl" title="Woof">
            <PawPrint className="h-5 w-5 text-primary-foreground" aria-hidden="true" />
          </span>
        </div>
      </header>

      <main id="main-content" className="mx-auto max-w-lg px-4 py-8">
        {currentStep === 1 && <OwnerInfoStep onComplete={handleOwnerComplete} initialData={ownerData} />}
        {currentStep === 2 && <PetInfoStep onComplete={handlePetComplete} initialData={petData} />}
        {currentStep === 3 && <QuizStep onComplete={handleQuizComplete} isLoading={isLoading} />}

        {error && currentStep === 3 && (
          <div role="alert" aria-live="polite" className="mt-5 rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm leading-relaxed text-destructive">
            {error}
          </div>
        )}
      </main>
    </div>
  )
}
