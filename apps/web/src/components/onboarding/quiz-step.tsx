'use client';

import { useState } from 'react';
import { Loader2, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import type { QuizAnswers } from '@/lib/api/quiz';

interface QuizStepProps {
  onComplete: (data: QuizAnswers) => void | Promise<void>;
  isLoading?: boolean;
}

const questions = [
  {
    id: 'activity_level',
    question: 'How active is your typical week with your pet?',
    type: 'single',
    options: [
      'Low - mostly relaxed activities',
      'Medium - a balanced mix',
      'High - very active lifestyle',
    ],
  },
  {
    id: 'schedule',
    question: 'When are you usually available for shared activities?',
    type: 'multiple',
    options: ['Weekday mornings', 'Weekday afternoons', 'Weekday evenings', 'Weekends'],
  },
  {
    id: 'walk_frequency',
    question: 'How often do you usually walk your pet?',
    type: 'single',
    options: ['Once a day', 'Twice a day', 'Three or more times a day', 'A few times a week'],
  },
  {
    id: 'socialization',
    question: 'How does your pet usually approach other animals?',
    type: 'single',
    options: [
      'Very social - enjoys most animals',
      'Selective - prefers certain play styles',
      'Cautious - needs slow introductions',
      'Prefers more space',
    ],
  },
  {
    id: 'environment',
    question: 'Which environment best describes your routine?',
    type: 'single',
    options: [
      'House with yard',
      'Apartment near parks',
      'Condo or townhouse',
      'Rural or farm setting',
    ],
  },
  {
    id: 'interests',
    question: 'Which activities would you actually want to plan with another owner?',
    type: 'multiple',
    options: [
      'Dog park visits',
      'Hiking trails',
      'Beach outings',
      'Training classes',
      'Pet-friendly cafes',
      'One-on-one playdates',
    ],
  },
  {
    id: 'experience',
    question: 'How much pet-handling experience do you have?',
    type: 'single',
    options: ['First-time owner', 'Some experience', 'Very experienced', 'Professional experience'],
  },
  {
    id: 'group_size',
    question: 'What group size feels comfortable?',
    type: 'single',
    options: [
      'One-on-one',
      'Small groups (2-3 pets)',
      'Medium groups (4-6 pets)',
      'Large groups (7+ pets)',
    ],
  },
  {
    id: 'distance',
    question: 'How far would you normally travel for a good meetup?',
    type: 'single',
    options: ['Within 1 mile', 'Within 3 miles', 'Within 5 miles', 'Within 10+ miles'],
  },
  {
    id: 'commitment',
    question: 'How often would you ideally meet compatible owners?',
    type: 'single',
    options: [
      'Daily',
      'A few times a week',
      'Weekly',
      'Every other week',
      'Monthly',
      'Flexible or spontaneous',
    ],
  },
] as const;

export function QuizStep({ onComplete, isLoading = false }: QuizStepProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState<QuizAnswers>({});

  const question = questions[currentQuestion];
  const isLastQuestion = currentQuestion === questions.length - 1;
  const progress = ((currentQuestion + 1) / questions.length) * 100;
  const currentAnswer = answers[question.id];
  const isAnswered =
    question.type === 'multiple'
      ? Array.isArray(currentAnswer) && currentAnswer.length > 0
      : typeof currentAnswer === 'string' && currentAnswer.length > 0;

  const handleAnswer = (value: string | string[]) => {
    setAnswers((current) => ({ ...current, [question.id]: value }));
  };

  const handleNext = async () => {
    if (!isAnswered || isLoading) return;
    if (isLastQuestion) {
      await onComplete(answers);
    } else {
      setCurrentQuestion((index) => index + 1);
    }
  };

  const handleBack = () => {
    if (!isLoading && currentQuestion > 0) {
      setCurrentQuestion((index) => index - 1);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-primary">
          <Sparkles className="h-5 w-5" aria-hidden="true" />
          <span className="text-sm font-semibold">Matching preferences</span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-balance">
          Add the context a profile cannot capture
        </h1>
        <p className="text-sm leading-relaxed text-muted-foreground">
          These answers are stored as a separate preference session so future ranking experiments
          can distinguish durable pet traits from owner intent.
        </p>
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Question {currentQuestion + 1} of {questions.length}
          </span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div
          className="h-2 overflow-hidden rounded-full bg-muted"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress)}
          aria-label="Matching preference quiz progress"
        >
          <div
            className="h-full bg-primary transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      <Card className="glass space-y-6 rounded-2xl p-5 sm:p-6">
        <h2 className="text-xl font-semibold leading-snug text-balance">{question.question}</h2>

        {question.type === 'single' ? (
          <RadioGroup
            value={typeof currentAnswer === 'string' ? currentAnswer : ''}
            onValueChange={handleAnswer}
            disabled={isLoading}
          >
            <div className="space-y-3">
              {question.options.map((option, index) => {
                const id = `${question.id}-${index}`;
                return (
                  <Label
                    key={option}
                    htmlFor={id}
                    className="flex min-h-14 cursor-pointer items-center gap-3 rounded-xl border border-border p-4 transition-colors hover:border-primary/40 hover:bg-primary/[0.035]"
                  >
                    <RadioGroupItem value={option} id={id} />
                    <span className="flex-1 leading-relaxed">{option}</span>
                  </Label>
                );
              })}
            </div>
          </RadioGroup>
        ) : (
          <div className="space-y-3">
            {question.options.map((option, index) => {
              const selected = Array.isArray(currentAnswer) ? currentAnswer : [];
              const isChecked = selected.includes(option);
              const id = `${question.id}-${index}`;

              return (
                <Label
                  key={option}
                  htmlFor={id}
                  className="flex min-h-14 cursor-pointer items-center gap-3 rounded-xl border border-border p-4 transition-colors hover:border-primary/40 hover:bg-primary/[0.035]"
                >
                  <Checkbox
                    id={id}
                    checked={isChecked}
                    disabled={isLoading}
                    onCheckedChange={(checked) => {
                      handleAnswer(
                        checked
                          ? [...selected, option]
                          : selected.filter((value) => value !== option)
                      );
                    }}
                  />
                  <span className="flex-1 leading-relaxed">{option}</span>
                </Label>
              );
            })}
          </div>
        )}
      </Card>

      <div className="flex gap-3">
        {currentQuestion > 0 && (
          <Button
            type="button"
            variant="outline"
            onClick={handleBack}
            disabled={isLoading}
            className="flex-1 bg-transparent"
          >
            Back
          </Button>
        )}
        <Button
          type="button"
          onClick={handleNext}
          disabled={!isAnswered || isLoading}
          className="flex-1"
        >
          {isLoading && isLastQuestion ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
              Creating Woof profile…
            </>
          ) : isLastQuestion ? (
            'Complete profile'
          ) : (
            'Next'
          )}
        </Button>
      </div>
    </div>
  );
}
