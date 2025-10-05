'use client';

import { useRouter } from 'next/navigation';
import { OnboardingQuiz } from '@/components/quiz/OnboardingQuiz';
import { QuizSession } from '@/types/quiz';
import { useSubmitQuiz } from '@/lib/api/hooks';
import { useUIStore } from '@/store/ui';
import { quizToFeatureVector } from '@/lib/ml/compatibilityScorer';

export default function QuizPage() {
  const router = useRouter();
  const { showToast } = useUIStore();
  const submitQuizMutation = useSubmitQuiz();

  const handleQuizComplete = async (session: QuizSession) => {
    try {
      // Convert quiz to ML feature vector
      const featureVector = quizToFeatureVector(session);

      // Submit to backend
      await submitQuizMutation.mutateAsync({
        session,
        featureVector,
      });

      showToast({
        message: 'Profile complete! Finding your perfect matches...',
        type: 'success',
      });

      // Redirect to feed/matches
      setTimeout(() => {
        router.push('/');
      }, 1500);
    } catch (error) {
      console.error('Failed to submit quiz:', error);
      showToast({
        message: 'Failed to save quiz responses. Please try again.',
        type: 'error',
      });
    }
  };

  const handleSkip = () => {
    // Allow skipping but show warning
    const confirmed = confirm(
      'Skipping the quiz will limit match quality. We recommend completing it for best results. Skip anyway?'
    );

    if (confirmed) {
      router.push('/');
    }
  };

  return (
    <div>
      <OnboardingQuiz onComplete={handleQuizComplete} onSkip={handleSkip} />
    </div>
  );
}
