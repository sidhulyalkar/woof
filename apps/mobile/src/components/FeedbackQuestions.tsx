import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme/tokens';

export interface Question {
  id: number;
  text: string;
  type: 'yes_no' | 'scale' | 'multiple_choice';
  options?: string[];
  importance: number;
}

interface FeedbackQuestionsProps {
  questions: Question[];
  onComplete: (responses: Record<number, any>) => void;
  onSkip: () => void;
  petName?: string;
}

export const FeedbackQuestions: React.FC<FeedbackQuestionsProps> = ({
  questions,
  onComplete,
  onSkip,
  petName = 'your pet',
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [responses, setResponses] = useState<Record<number, any>>({});
  const [fadeAnim] = useState(new Animated.Value(1));

  const currentQuestion = questions[currentIndex];
  const progress = ((currentIndex + 1) / questions.length) * 100;

  const handleResponse = (questionId: number, response: any) => {
    const newResponses = { ...responses, [questionId]: response };
    setResponses(newResponses);

    // Animate to next question
    Animated.sequence([
      Animated.timing(fadeAnim, {
        toValue: 0,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();

    setTimeout(() => {
      if (currentIndex < questions.length - 1) {
        setCurrentIndex(currentIndex + 1);
      } else {
        onComplete(newResponses);
      }
    }, 200);
  };

  const handleBack = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const renderYesNoQuestion = () => (
    <View style={styles.optionsContainer}>
      <TouchableOpacity
        style={[styles.yesNoButton, styles.yesButton]}
        onPress={() => handleResponse(currentQuestion.id, true)}
      >
        <Ionicons name="checkmark-circle" size={32} color="#10b981" />
        <Text style={styles.yesNoText}>Yes</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.yesNoButton, styles.noButton]}
        onPress={() => handleResponse(currentQuestion.id, false)}
      >
        <Ionicons name="close-circle" size={32} color="#ef4444" />
        <Text style={styles.yesNoText}>No</Text>
      </TouchableOpacity>
    </View>
  );

  const renderScaleQuestion = () => {
    const scale = [1, 2, 3, 4, 5];
    return (
      <View style={styles.scaleContainer}>
        <View style={styles.scaleLabels}>
          <Text style={styles.scaleLabel}>Not at all</Text>
          <Text style={styles.scaleLabel}>Very much</Text>
        </View>
        <View style={styles.scaleButtons}>
          {scale.map((value) => (
            <TouchableOpacity
              key={value}
              style={styles.scaleButton}
              onPress={() => handleResponse(currentQuestion.id, value)}
            >
              <Text style={styles.scaleButtonText}>{value}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    );
  };

  const renderMultipleChoiceQuestion = () => (
    <View style={styles.multipleChoiceContainer}>
      {currentQuestion.options?.map((option, index) => (
        <TouchableOpacity
          key={index}
          style={styles.multipleChoiceButton}
          onPress={() => handleResponse(currentQuestion.id, option)}
        >
          <Text style={styles.multipleChoiceText}>{option}</Text>
          <Ionicons name="chevron-forward" size={20} color="#6b7280" />
        </TouchableOpacity>
      ))}
    </View>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Help us improve matches</Text>
        <TouchableOpacity onPress={onSkip} style={styles.skipButton}>
          <Text style={styles.skipText}>Skip</Text>
        </TouchableOpacity>
      </View>

      {/* Progress Bar */}
      <View style={styles.progressBarContainer}>
        <View style={[styles.progressBar, { width: `${progress}%` }]} />
      </View>
      <Text style={styles.progressText}>
        Question {currentIndex + 1} of {questions.length}
      </Text>

      {/* Question Content */}
      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        <Animated.View style={{ opacity: fadeAnim }}>
          {/* Importance Indicator */}
          {currentQuestion.importance > 0.7 && (
            <View style={styles.importanceBadge}>
              <Ionicons name="star" size={16} color="#f59e0b" />
              <Text style={styles.importanceText}>High Priority</Text>
            </View>
          )}

          {/* Question Text */}
          <Text style={styles.questionText}>{currentQuestion.text}</Text>

          {/* Context */}
          <Text style={styles.contextText}>
            This helps us find better matches for {petName}
          </Text>

          {/* Answer Options */}
          <View style={styles.answersContainer}>
            {currentQuestion.type === 'yes_no' && renderYesNoQuestion()}
            {currentQuestion.type === 'scale' && renderScaleQuestion()}
            {currentQuestion.type === 'multiple_choice' &&
              renderMultipleChoiceQuestion()}
          </View>
        </Animated.View>
      </ScrollView>

      {/* Navigation */}
      <View style={styles.navigation}>
        {currentIndex > 0 && (
          <TouchableOpacity onPress={handleBack} style={styles.backButton}>
            <Ionicons name="arrow-back" size={20} color="#8b5cf6" />
            <Text style={styles.backText}>Back</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: theme.spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#111827',
  },
  skipButton: {
    padding: theme.spacing.sm,
  },
  skipText: {
    fontSize: 16,
    color: '#6b7280',
    fontWeight: '500',
  },
  progressBarContainer: {
    height: 4,
    backgroundColor: '#e5e7eb',
    borderRadius: 2,
    marginBottom: theme.spacing.sm,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#8b5cf6',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 14,
    color: '#6b7280',
    marginBottom: theme.spacing.xl,
  },
  content: {
    flex: 1,
  },
  importanceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef3c7',
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: 12,
    alignSelf: 'flex-start',
    marginBottom: theme.spacing.md,
  },
  importanceText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f59e0b',
    marginLeft: theme.spacing.xs,
  },
  questionText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#111827',
    marginBottom: theme.spacing.md,
    lineHeight: 28,
  },
  contextText: {
    fontSize: 15,
    color: '#6b7280',
    marginBottom: theme.spacing.xl,
    lineHeight: 22,
  },
  answersContainer: {
    marginTop: theme.spacing.lg,
  },
  optionsContainer: {
    flexDirection: 'row',
    gap: theme.spacing.md,
  },
  yesNoButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.lg,
    borderRadius: 16,
    gap: theme.spacing.sm,
  },
  yesButton: {
    backgroundColor: '#d1fae5',
    borderWidth: 2,
    borderColor: '#10b981',
  },
  noButton: {
    backgroundColor: '#fee2e2',
    borderWidth: 2,
    borderColor: '#ef4444',
  },
  yesNoText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
  },
  scaleContainer: {
    gap: theme.spacing.md,
  },
  scaleLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.sm,
  },
  scaleLabel: {
    fontSize: 13,
    color: '#6b7280',
  },
  scaleButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: theme.spacing.sm,
  },
  scaleButton: {
    flex: 1,
    aspectRatio: 1,
    backgroundColor: '#f3f4f6',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#e5e7eb',
  },
  scaleButtonText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#374151',
  },
  multipleChoiceContainer: {
    gap: theme.spacing.md,
  },
  multipleChoiceButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: theme.spacing.lg,
    backgroundColor: '#f9fafb',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  multipleChoiceText: {
    fontSize: 16,
    color: '#111827',
    fontWeight: '500',
  },
  navigation: {
    paddingTop: theme.spacing.lg,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  backText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#8b5cf6',
  },
});
