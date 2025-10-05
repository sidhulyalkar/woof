'use client';

import { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Check } from 'lucide-react';
import { QuizQuestion, QuizResponse, QuizSession } from '@/types/quiz';
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
  const currentSection = QUIZ_SECTIONS.find((s) => s.id === currentQuestion.category);

  // Check if current question is answered
  const currentResponse = responses.find((r) => r.questionId === currentQuestion.id);
  const isAnswered = currentResponse !== undefined;

  const handleAnswer = (answer: string | string[] | number, customAnswer?: string) => {
    const newResponse: QuizResponse = {
      questionId: currentQuestion.id,
      answer,
      customAnswer,
      timestamp: new Date().toISOString(),
    };

    setResponses((prev) => {
      const filtered = prev.filter((r) => r.questionId !== currentQuestion.id);
      return [...filtered, newResponse];
    });
  };

  const handleNext = () => {
    if (currentQuestion.required && !isAnswered) {
      showToast({ message: 'Please answer this question to continue', type: 'error' });
      return;
    }

    if (currentQuestionIndex < QUIZ_QUESTIONS.length - 1) {
      setCurrentQuestionIndex((prev) => prev + 1);
    } else {
      handleComplete();
    }
  };

  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex((prev) => prev - 1);
    }
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 pb-20">
      {/* Header with Progress */}
      <div className="sticky top-0 z-50 bg-white/80 backdrop-blur-lg border-b border-gray-200">
        <div className="max-w-2xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-3">
            <button
              onClick={handleBack}
              disabled={currentQuestionIndex === 0}
              className="p-2 rounded-full hover:bg-gray-100 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              <ChevronLeft className="h-5 w-5 text-gray-700" />
            </button>
            <div className="text-center flex-1">
              <div className="text-sm font-semibold text-gray-900">
                Question {currentQuestionIndex + 1} of {QUIZ_QUESTIONS.length}
              </div>
              {currentSection && (
                <div className="text-xs text-gray-600 flex items-center justify-center gap-1 mt-1">
                  <span>{currentSection.icon}</span>
                  <span>{currentSection.title}</span>
                </div>
              )}
            </div>
            {onSkip && (
              <button
                onClick={onSkip}
                className="text-sm text-gray-600 hover:text-gray-900 px-3 py-1 rounded-full hover:bg-gray-100"
              >
                Skip for now
              </button>
            )}
          </div>

          {/* Progress Bar */}
          <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Section Header */}
      {currentSection && currentQuestionIndex === QUIZ_QUESTIONS.findIndex((q) => q.category === currentSection.id) && (
        <div className="max-w-2xl mx-auto px-4 py-8 text-center">
          <div className="text-5xl mb-3">{currentSection.icon}</div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">{currentSection.title}</h2>
          <p className="text-gray-600">{currentSection.description}</p>
        </div>
      )}

      {/* Question Card */}
      <div className="max-w-2xl mx-auto px-4 py-6">
        <QuizQuestionCard
          question={currentQuestion}
          value={currentResponse?.answer}
          customValue={currentResponse?.customAnswer}
          onChange={handleAnswer}
        />
      </div>

      {/* Navigation Footer */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/80 backdrop-blur-lg border-t border-gray-200 p-4">
        <div className="max-w-2xl mx-auto flex gap-3">
          <button
            onClick={handleBack}
            disabled={currentQuestionIndex === 0}
            className="px-6 py-3 rounded-full border-2 border-gray-300 text-gray-700 font-semibold hover:border-gray-400 hover:bg-gray-50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
          >
            Back
          </button>
          <button
            onClick={handleNext}
            disabled={currentQuestion.required && !isAnswered}
            className="flex-1 px-6 py-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 text-white font-semibold hover:from-blue-600 hover:to-purple-600 disabled:opacity-30 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
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
