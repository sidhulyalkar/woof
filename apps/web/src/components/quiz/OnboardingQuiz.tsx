'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Check } from 'lucide-react';
import type { QuizResponse, QuizSession } from '@/types/quiz';
import { QUIZ_QUESTIONS, QUIZ_SECTIONS } from '@/data/quizQuestions';
import { QuizQuestionCard } from './QuizQuestionCard';
import { useUIStore } from '@/store/ui';
import { useSessionStore } from '@/store/session';

interface OnboardingQuizProps {
  petId?: string;
  onComplete: (session: QuizSession) => void;
  onSkip?: () => void;
}

export function OnboardingQuiz({ petId, onComplete, onSkip }: OnboardingQuizProps) {
  const { user } = useSessionStore();
  const { showToast } = useUIStore();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [responses, setResponses] = useState<QuizResponse[]>([]);
  const [sessionId] = useState(`quiz_${Date.now()}_${user?.id}`);

  const currentQuestion = QUIZ_QUESTIONS[currentQuestionIndex];
  const progress = ((currentQuestionIndex + 1) / QUIZ_QUESTIONS.length) * 100;
  const currentSection = QUIZ_SECTIONS.find((section) => section.id === currentQuestion.sectionId);

  const currentResponse = responses.find((response) => response.questionId === currentQuestion.id);
  const isAnswered = currentResponse !== undefined;

  const handleAnswer = (answer: string | string[] | number, customAnswer?: string) => {
    const newResponse: QuizResponse = {
      questionId: currentQuestion.id,
      answer,
      customAnswer,
      timestamp: new Date().toISOString(),
    };

    setResponses((previous) => [
      ...previous.filter((response) => response.questionId !== currentQuestion.id),
      newResponse,
    ]);
  };

  const handleComplete = () => {
    const session: QuizSession = {
      id: sessionId,
      userId: user?.id || '',
      petId,
      responses,
      completedAt: new Date().toISOString(),
      currentStep: QUIZ_QUESTIONS.length,
      totalSteps: QUIZ_QUESTIONS.length,
    };

    showToast({ message: 'Quiz completed! 🎉', type: 'success' });
    onComplete(session);
  };

  const handleNext = () => {
    if (currentQuestion.required && !isAnswered) {
      showToast({ message: 'Please answer this question to continue', type: 'error' });
      return;
    }

    if (currentQuestionIndex < QUIZ_QUESTIONS.length - 1) {
      setCurrentQuestionIndex((previous) => previous + 1);
      return;
    }

    handleComplete();
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((previous) => previous - 1);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 pb-20">
      <div className="sticky top-0 z-50 border-b border-gray-200 bg-white/80 backdrop-blur-lg">
        <div className="mx-auto max-w-2xl px-4 py-4">
          <div className="mb-3 flex items-center justify-between">
            <button
              onClick={handleBack}
              disabled={currentQuestionIndex === 0}
              className="rounded-full p-2 transition-all hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-30"
            >
              <ChevronLeft className="h-5 w-5 text-gray-700" />
            </button>
            <div className="flex-1 text-center">
              <div className="text-sm font-semibold text-gray-900">
                Question {currentQuestionIndex + 1} of {QUIZ_QUESTIONS.length}
              </div>
              {currentSection && (
                <div className="mt-1 flex items-center justify-center gap-1 text-xs text-gray-600">
                  <span>{currentSection.icon}</span>
                  <span>{currentSection.title}</span>
                </div>
              )}
            </div>
            {onSkip && (
              <button
                onClick={onSkip}
                className="rounded-full px-3 py-1 text-sm text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              >
                Skip for now
              </button>
            )}
          </div>

          <div className="relative h-2 overflow-hidden rounded-full bg-gray-200">
            <div
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {currentSection &&
        currentQuestionIndex ===
          QUIZ_QUESTIONS.findIndex((question) => question.sectionId === currentSection.id) && (
          <div className="mx-auto max-w-2xl px-4 py-8 text-center">
            <div className="mb-3 text-5xl">{currentSection.icon}</div>
            <h2 className="mb-2 text-2xl font-bold text-gray-900">{currentSection.title}</h2>
            <p className="text-gray-600">{currentSection.description}</p>
          </div>
        )}

      <div className="mx-auto max-w-2xl px-4 py-6">
        <QuizQuestionCard
          question={currentQuestion}
          value={currentResponse?.answer}
          customValue={currentResponse?.customAnswer}
          onChange={handleAnswer}
        />
      </div>

      <div className="fixed bottom-0 left-0 right-0 border-t border-gray-200 bg-white/80 p-4 backdrop-blur-lg">
        <div className="mx-auto flex max-w-2xl gap-3">
          <button
            onClick={handleBack}
            disabled={currentQuestionIndex === 0}
            className="rounded-full border-2 border-gray-300 px-6 py-3 font-semibold text-gray-700 transition-all hover:border-gray-400 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-30"
          >
            Back
          </button>
          <button
            onClick={handleNext}
            disabled={currentQuestion.required && !isAnswered}
            className="flex flex-1 items-center justify-center gap-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 px-6 py-3 font-semibold text-white transition-all hover:from-blue-600 hover:to-purple-600 disabled:cursor-not-allowed disabled:opacity-30"
          >
            {currentQuestionIndex === QUIZ_QUESTIONS.length - 1 ? (
              <>
                <Check className="h-5 w-5" />
                Complete Quiz
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-5 w-5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
