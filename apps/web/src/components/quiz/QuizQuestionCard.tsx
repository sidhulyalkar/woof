'use client';

import { useState } from 'react';
import { QuizQuestion } from '@/types/quiz';
import { Check, Edit3 } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';

interface QuizQuestionCardProps {
  question: QuizQuestion;
  value?: string | string[] | number;
  customValue?: string;
  onChange: (answer: string | string[] | number, customAnswer?: string) => void;
}

export function QuizQuestionCard({ question, value, customValue, onChange }: QuizQuestionCardProps) {
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customText, setCustomText] = useState(customValue || '');

  const handleMultipleChoice = (optionValue: string) => {
    onChange(optionValue, customText || undefined);
  };

  const handleMultipleSelect = (optionValue: string) => {
    const currentValues = Array.isArray(value) ? value : [];
    const newValues = currentValues.includes(optionValue)
      ? currentValues.filter((v) => v !== optionValue)
      : [...currentValues, optionValue];
    onChange(newValues, customText || undefined);
  };

  const handleScale = (scaleValue: number) => {
    onChange(scaleValue);
  };

  const handleText = (text: string) => {
    onChange(text);
  };

  const handleCustomSubmit = () => {
    if (customText.trim()) {
      onChange(customText, customText);
      setShowCustomInput(false);
    }
  };

  return (
    <div className="bg-white rounded-3xl shadow-lg p-8 border border-gray-100">
      {/* Question */}
      <div className="mb-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-2">{question.question}</h3>
        {question.description && <p className="text-gray-600 text-sm">{question.description}</p>}
        {question.required && <span className="text-red-500 text-xs mt-1 block">* Required</span>}
      </div>

      {/* Multiple Choice */}
      {question.type === 'multiple_choice' && question.options && (
        <div className="space-y-3">
          {question.options.map((option) => (
            <button
              key={option.id}
              onClick={() => handleMultipleChoice(option.value.toString())}
              className={cn(
                'w-full p-4 rounded-2xl border-2 text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]',
                value === option.value.toString()
                  ? 'border-blue-500 bg-blue-50 shadow-md'
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="font-semibold text-gray-900">{option.label}</div>
                  {option.description && <div className="text-sm text-gray-600 mt-1">{option.description}</div>}
                </div>
                {value === option.value.toString() && (
                  <div className="w-6 h-6 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                    <Check className="h-4 w-4 text-white" />
                  </div>
                )}
              </div>
            </button>
          ))}

          {question.allowCustom && (
            <div className="pt-2">
              {!showCustomInput ? (
                <button
                  onClick={() => setShowCustomInput(true)}
                  className="w-full p-4 rounded-2xl border-2 border-dashed border-gray-300 hover:border-blue-400 hover:bg-blue-50 text-gray-600 hover:text-blue-600 font-semibold transition-all flex items-center justify-center gap-2"
                >
                  <Edit3 className="h-4 w-4" />
                  Other (specify)
                </button>
              ) : (
                <div className="flex gap-2">
                  <Input
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    placeholder="Type your answer..."
                    className="flex-1"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleCustomSubmit();
                      }
                    }}
                  />
                  <button
                    onClick={handleCustomSubmit}
                    disabled={!customText.trim()}
                    className="px-4 py-2 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-30 transition-all"
                  >
                    Add
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Multiple Select */}
      {question.type === 'multiple_select' && question.options && (
        <div className="space-y-3">
          {question.options.map((option) => {
            const isSelected = Array.isArray(value) && value.includes(option.value.toString());
            return (
              <button
                key={option.id}
                onClick={() => handleMultipleSelect(option.value.toString())}
                className={cn(
                  'w-full p-4 rounded-2xl border-2 text-left transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]',
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                )}
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="font-semibold text-gray-900">{option.label}</div>
                  <div
                    className={cn(
                      'w-6 h-6 rounded-lg border-2 flex items-center justify-center transition-all',
                      isSelected ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                    )}
                  >
                    {isSelected && <Check className="h-4 w-4 text-white" />}
                  </div>
                </div>
              </button>
            );
          })}

          {question.allowCustom && (
            <div className="pt-2">
              {!showCustomInput ? (
                <button
                  onClick={() => setShowCustomInput(true)}
                  className="w-full p-4 rounded-2xl border-2 border-dashed border-gray-300 hover:border-blue-400 hover:bg-blue-50 text-gray-600 hover:text-blue-600 font-semibold transition-all flex items-center justify-center gap-2"
                >
                  <Edit3 className="h-4 w-4" />
                  Add custom option
                </button>
              ) : (
                <div className="flex gap-2">
                  <Input
                    value={customText}
                    onChange={(e) => setCustomText(e.target.value)}
                    placeholder="Type your custom option..."
                    className="flex-1"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleCustomSubmit();
                      }
                    }}
                  />
                  <button
                    onClick={handleCustomSubmit}
                    disabled={!customText.trim()}
                    className="px-4 py-2 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-30 transition-all"
                  >
                    Add
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Scale */}
      {question.type === 'scale' && question.scaleRange && (
        <div className="space-y-6">
          <div className="flex items-center justify-between text-sm text-gray-600 px-2">
            <span>{question.scaleRange.minLabel}</span>
            <span>{question.scaleRange.maxLabel}</span>
          </div>

          <div className="flex items-center justify-between gap-2">
            {Array.from(
              { length: question.scaleRange.max - question.scaleRange.min + 1 },
              (_, i) => i + question.scaleRange!.min
            ).map((num) => (
              <button
                key={num}
                onClick={() => handleScale(num)}
                className={cn(
                  'flex-1 h-16 rounded-2xl border-2 font-bold text-lg transition-all duration-200 hover:scale-110 active:scale-95',
                  value === num
                    ? 'border-blue-500 bg-blue-500 text-white shadow-lg scale-110'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
                )}
              >
                {num}
              </button>
            ))}
          </div>

          {value !== undefined && (
            <div className="text-center">
              <div className="inline-block px-6 py-2 bg-blue-50 rounded-full">
                <span className="text-sm text-gray-600">Selected: </span>
                <span className="font-bold text-blue-600">{value}</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Text Input */}
      {question.type === 'text' && (
        <div>
          <Input
            value={(value as string) || ''}
            onChange={(e) => handleText(e.target.value)}
            placeholder="Type your answer..."
            className="w-full p-4 text-lg"
          />
        </div>
      )}
    </div>
  );
}
