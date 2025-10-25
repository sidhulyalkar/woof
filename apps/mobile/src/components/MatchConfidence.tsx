import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { theme } from '../theme/tokens';

interface MatchConfidenceProps {
  score: number; // 0-1
  confidence: number; // 0-1
  uncertainty: number; // 0-1
  modelWeights?: {
    gat: number;
    simgnn: number;
    diffusion: number;
  };
  onRequestFeedback?: () => void;
  showDetails?: boolean;
}

export const MatchConfidence: React.FC<MatchConfidenceProps> = ({
  score,
  confidence,
  uncertainty,
  modelWeights,
  onRequestFeedback,
  showDetails = false,
}) => {
  const getScoreColor = (value: number) => {
    if (value >= 0.8) return '#10b981';
    if (value >= 0.6) return '#f59e0b';
    return '#ef4444';
  };

  const getConfidenceLabel = (conf: number) => {
    if (conf >= 0.8) return 'Very Confident';
    if (conf >= 0.6) return 'Confident';
    if (conf >= 0.4) return 'Somewhat Confident';
    return 'Uncertain';
  };

  const scoreColor = getScoreColor(score);
  const shouldShowFeedback = uncertainty > 0.5 || confidence < 0.6;

  return (
    <View style={styles.container}>
      {/* Match Score */}
      <View style={styles.scoreContainer}>
        <View style={styles.scoreCircle}>
          <Text style={[styles.scoreValue, { color: scoreColor }]}>
            {Math.round(score * 100)}
          </Text>
          <Text style={styles.scoreLabel}>Match</Text>
        </View>

        <View style={styles.scoreDetails}>
          <Text style={styles.scoreTitle}>Compatibility Score</Text>
          <Text style={styles.scoreSubtitle}>
            {getConfidenceLabel(confidence)}
          </Text>
        </View>
      </View>

      {/* Confidence Bar */}
      <View style={styles.confidenceContainer}>
        <View style={styles.confidenceHeader}>
          <Ionicons
            name={confidence >= 0.7 ? 'shield-checkmark' : 'alert-circle'}
            size={16}
            color={confidence >= 0.7 ? '#10b981' : '#f59e0b'}
          />
          <Text style={styles.confidenceText}>
            Prediction Confidence: {Math.round(confidence * 100)}%
          </Text>
        </View>

        <View style={styles.confidenceBar}>
          <View
            style={[
              styles.confidenceFill,
              {
                width: `${confidence * 100}%`,
                backgroundColor: confidence >= 0.7 ? '#10b981' : '#f59e0b',
              },
            ]}
          />
        </View>
      </View>

      {/* Model Breakdown (if showing details) */}
      {showDetails && modelWeights && (
        <View style={styles.modelBreakdown}>
          <Text style={styles.breakdownTitle}>Model Contributions</Text>
          {Object.entries(modelWeights).map(([model, weight]) => (
            <View key={model} style={styles.modelRow}>
              <Text style={styles.modelName}>{model.toUpperCase()}</Text>
              <View style={styles.modelWeightContainer}>
                <View style={styles.modelWeightBar}>
                  <View
                    style={[
                      styles.modelWeightFill,
                      { width: `${weight * 100}%` },
                    ]}
                  />
                </View>
                <Text style={styles.modelWeightText}>
                  {Math.round(weight * 100)}%
                </Text>
              </View>
            </View>
          ))}
        </View>
      )}

      {/* Feedback Request */}
      {shouldShowFeedback && onRequestFeedback && (
        <TouchableOpacity
          style={styles.feedbackButton}
          onPress={onRequestFeedback}
        >
          <Ionicons name="help-circle-outline" size={20} color="#8b5cf6" />
          <Text style={styles.feedbackText}>
            Help improve this match
          </Text>
          <Ionicons name="chevron-forward" size={20} color="#8b5cf6" />
        </TouchableOpacity>
      )}

      {/* Uncertainty Warning */}
      {uncertainty > 0.7 && (
        <View style={styles.warningContainer}>
          <Ionicons name="information-circle" size={20} color="#f59e0b" />
          <Text style={styles.warningText}>
            We need more information to make a confident prediction
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: theme.spacing.lg,
    ...theme.shadows.md,
  },
  scoreContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.lg,
  },
  scoreCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#f9fafb',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: theme.spacing.md,
  },
  scoreValue: {
    fontSize: 28,
    fontWeight: '700',
  },
  scoreLabel: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  scoreDetails: {
    flex: 1,
  },
  scoreTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#111827',
    marginBottom: 4,
  },
  scoreSubtitle: {
    fontSize: 14,
    color: '#6b7280',
  },
  confidenceContainer: {
    marginBottom: theme.spacing.lg,
  },
  confidenceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.xs,
    marginBottom: theme.spacing.sm,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e5e7eb',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  modelBreakdown: {
    paddingTop: theme.spacing.lg,
    borderTopWidth: 1,
    borderTopColor: '#e5e7eb',
    marginBottom: theme.spacing.lg,
  },
  breakdownTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#111827',
    marginBottom: theme.spacing.md,
  },
  modelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.sm,
  },
  modelName: {
    fontSize: 13,
    fontWeight: '500',
    color: '#6b7280',
    width: 80,
  },
  modelWeightContainer: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.spacing.sm,
  },
  modelWeightBar: {
    flex: 1,
    height: 6,
    backgroundColor: '#e5e7eb',
    borderRadius: 3,
    overflow: 'hidden',
  },
  modelWeightFill: {
    height: '100%',
    backgroundColor: '#8b5cf6',
    borderRadius: 3,
  },
  modelWeightText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#374151',
    width: 40,
    textAlign: 'right',
  },
  feedbackButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: theme.spacing.md,
    backgroundColor: '#f3e8ff',
    borderRadius: 12,
    gap: theme.spacing.sm,
    marginBottom: theme.spacing.md,
  },
  feedbackText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#8b5cf6',
  },
  warningContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: theme.spacing.md,
    backgroundColor: '#fef3c7',
    borderRadius: 12,
    gap: theme.spacing.sm,
  },
  warningText: {
    flex: 1,
    fontSize: 13,
    color: '#92400e',
    lineHeight: 18,
  },
});
