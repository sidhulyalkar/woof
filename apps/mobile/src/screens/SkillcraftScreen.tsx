import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  socialAdventureApi,
  type ArcadeAttempt,
  type ArcadeCatalog,
  type ArcadeChallengeKey,
  type ArcadeReceipt,
  type ArcadeScenario,
} from '../api/social-adventure';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Skillcraft'>;

type ShareState = 'idle' | 'pending' | 'shared' | 'error';

const challengeIcon: Record<ArcadeChallengeKey, keyof typeof Ionicons.glyphMap> = {
  MAKE_IT_EASIER: 'options-outline',
  CATCH_THE_GOOD: 'sparkles-outline',
  PAIRING_LAB: 'link-outline',
  MARKER_TIMING: 'timer-outline',
};

const challengeRoom: Record<ArcadeChallengeKey, string> = {
  MAKE_IT_EASIER: 'Setup Lab',
  CATCH_THE_GOOD: 'Observation Room',
  PAIRING_LAB: 'Association Studio',
  MARKER_TIMING: 'Timing Deck',
};

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

export default function SkillcraftScreen({ navigation }: Props) {
  const [catalog, setCatalog] = useState<ArcadeCatalog | null>(null);
  const [attempt, setAttempt] = useState<ArcadeAttempt | null>(null);
  const [receipt, setReceipt] = useState<ArcadeReceipt | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [startingKey, setStartingKey] = useState<ArcadeChallengeKey | null>(null);
  const [completing, setCompleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shareState, setShareState] = useState<ShareState>('idle');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [timingExpired, setTimingExpired] = useState(false);
  const timingStartRef = useRef<number | null>(null);

  const loadCatalog = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      setCatalog(await socialAdventureApi.arcade());
      setError(null);
    } catch {
      setError('Skillcraft is unavailable right now. No practice result or social score changed.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void loadCatalog();
    }, [loadCatalog])
  );

  useEffect(() => {
    const timing = attempt?.scenario.timing;
    if (!timing || receipt || timingExpired || timingStartRef.current === null) return;

    const timer = setInterval(() => {
      if (timingStartRef.current === null) return;
      const elapsed = Math.max(0, Date.now() - timingStartRef.current);
      const bounded = Math.min(timing.durationMs, elapsed);
      setElapsedMs(bounded);
      if (bounded >= timing.durationMs) setTimingExpired(true);
    }, 40);

    return () => clearInterval(timer);
  }, [attempt, receipt, timingExpired]);

  const startRound = async (challenge: ArcadeScenario) => {
    if (startingKey || completing) return;
    setStartingKey(challenge.challengeKey);
    setError(null);
    setShareState('idle');
    try {
      const nextAttempt = await socialAdventureApi.startArcadeAttempt(challenge.challengeKey);
      setAttempt(nextAttempt);
      setReceipt(null);
      setElapsedMs(0);
      setTimingExpired(false);
      timingStartRef.current = nextAttempt.scenario.timing ? Date.now() : null;
    } catch {
      setError('That Skillcraft round could not start. Nothing was scored.');
    } finally {
      setStartingKey(null);
    }
  };

  const completeRound = async (response: Record<string, unknown>) => {
    if (!attempt || completing || receipt) return;
    setCompleting(true);
    setError(null);
    try {
      const nextReceipt = await socialAdventureApi.completeArcadeAttempt(
        attempt.attemptId,
        response
      );
      setReceipt(nextReceipt);
      timingStartRef.current = null;
      void socialAdventureApi
        .arcade()
        .then((nextCatalog) => setCatalog(nextCatalog))
        .catch(() => undefined);
    } catch {
      setError(
        'Woof could not score that practice round. Start a fresh round before trying again.'
      );
    } finally {
      setCompleting(false);
    }
  };

  const markTiming = () => {
    if (!attempt?.scenario.timing || timingStartRef.current === null || timingExpired) return;
    const tapMs = Math.max(0, Date.now() - timingStartRef.current);
    void completeRound({ tapMs });
  };

  const resetRound = () => {
    setAttempt(null);
    setReceipt(null);
    setElapsedMs(0);
    setTimingExpired(false);
    setShareState('idle');
    timingStartRef.current = null;
    setError(null);
  };

  const shareResult = async () => {
    if (!receipt || shareState === 'pending' || shareState === 'shared') return;
    setShareState('pending');
    try {
      await socialAdventureApi.shareSkillAttempt(receipt.attemptId);
      setShareState('shared');
    } catch {
      setShareState('error');
    }
  };

  const completedThisWeek =
    catalog?.challenges.filter((challenge) => challenge.bestScore !== null).length ?? 0;
  const totalChallenges = catalog?.challenges.length ?? 0;
  const breadthProgress = clampPercent((completedThisWeek / Math.max(1, totalChallenges)) * 100);
  const timing = attempt?.scenario.timing;
  const timingProgress = timing ? clampPercent((elapsedMs / timing.durationMs) * 100) : 0;

  if (loading && !catalog) {
    return (
      <View style={styles.centered} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening Skillcraft…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => void loadCatalog(true)} />
      }
    >
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="game-controller-outline" size={24} color={colors.primary[700]} />
        </View>
        <Text style={styles.eyebrow}>HUMAN SKILL ARCADE</Text>
        <Text style={styles.title}>Skillcraft</Text>
        <Text style={styles.subtitle}>
          Practice the human mechanics behind better dog days: setup, observation, positive
          association, and timing. The dog does not have to perform for you to play.
        </Text>

        {catalog && (
          <View style={styles.weekPanel}>
            <View style={styles.weekHeader}>
              <View>
                <Text style={styles.weekTitle}>This week&apos;s rooms</Text>
                <Text style={styles.weekSubtitle}>Breadth counts once. Grinding does not.</Text>
              </View>
              <Text style={styles.weekCount}>
                {completedThisWeek}/{totalChallenges}
              </Text>
            </View>
            <View
              style={styles.progressTrack}
              accessibilityRole="progressbar"
              accessibilityValue={{ min: 0, max: 100, now: Math.round(breadthProgress) }}
            >
              <View style={[styles.progressFill, { width: `${breadthProgress}%` }]} />
            </View>
            <Text style={styles.weekBoundary}>
              Practice scores stay personal feedback. Completing each different room once in the
              weekly season contributes one fixed breadth unit; retries and higher scores add no
              rank.
            </Text>
          </View>
        )}
      </View>

      {error && (
        <View style={styles.errorCard} accessibilityRole="alert">
          <Ionicons name="alert-circle-outline" size={19} color={colors.error.dark} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {!attempt && catalog && (
        <View style={styles.challengeList}>
          {catalog.challenges.map((challenge) => {
            const practiced = challenge.bestScore !== null;
            const starting = startingKey === challenge.challengeKey;
            return (
              <View key={challenge.challengeKey} style={styles.challengeCard}>
                <View style={styles.challengeHeader}>
                  <View style={styles.challengeIcon}>
                    <Ionicons
                      name={challengeIcon[challenge.challengeKey]}
                      size={21}
                      color={colors.primary[700]}
                    />
                  </View>
                  <View style={styles.challengeCopy}>
                    <Text style={styles.roomLabel}>{challengeRoom[challenge.challengeKey]}</Text>
                    <Text style={styles.challengeTitle}>{challenge.title}</Text>
                    <Text style={styles.skillLabel}>{challenge.skill}</Text>
                  </View>
                  <View style={[styles.badge, practiced && styles.badgePracticed]}>
                    <Ionicons
                      name={practiced ? 'checkmark-circle' : 'sparkles-outline'}
                      size={13}
                      color={practiced ? colors.success.dark : colors.primary[700]}
                    />
                    <Text style={[styles.badgeText, practiced && styles.badgeTextPracticed]}>
                      {practiced ? 'Explored' : 'New'}
                    </Text>
                  </View>
                </View>

                <Text style={styles.challengePrompt}>{challenge.prompt}</Text>
                {practiced && (
                  <Text style={styles.personalBest}>
                    Personal practice best: {challenge.bestScore}/100
                  </Text>
                )}

                <Pressable
                  accessibilityRole="button"
                  disabled={startingKey !== null}
                  style={[styles.playButton, startingKey !== null && styles.disabled]}
                  onPress={() => void startRound(challenge)}
                >
                  {starting ? (
                    <ActivityIndicator color="#ffffff" />
                  ) : (
                    <>
                      <Ionicons name="play" size={16} color="#ffffff" />
                      <Text style={styles.playButtonText}>
                        {practiced ? 'Practice again' : 'Play room'}
                      </Text>
                    </>
                  )}
                </Pressable>
              </View>
            );
          })}

          <Text style={styles.scoringCopy}>{catalog.scoring}</Text>
        </View>
      )}

      {attempt && (
        <View style={styles.roundCard}>
          <View style={styles.roundHeader}>
            <View style={styles.roundIcon}>
              <Ionicons
                name={challengeIcon[attempt.scenario.challengeKey]}
                size={22}
                color="#ffffff"
              />
            </View>
            <View style={styles.roundCopy}>
              <Text style={styles.roundEyebrow}>{attempt.scenario.skill}</Text>
              <Text style={styles.roundTitle}>{attempt.scenario.title}</Text>
            </View>
          </View>

          <Text style={styles.roundPrompt}>{attempt.scenario.prompt}</Text>

          {!receipt && attempt.scenario.options && (
            <View style={styles.optionsList}>
              {attempt.scenario.options.map((option) => (
                <Pressable
                  key={option.id}
                  accessibilityRole="button"
                  disabled={completing}
                  style={[styles.optionButton, completing && styles.disabled]}
                  onPress={() => void completeRound({ optionId: option.id })}
                >
                  <Text style={styles.optionText}>{option.label}</Text>
                  <Ionicons name="chevron-forward" size={17} color={colors.gray[400]} />
                </Pressable>
              ))}
            </View>
          )}

          {!receipt && timing && (
            <View style={styles.timingPanel}>
              <View style={styles.timingTitleRow}>
                <Ionicons name="timer-outline" size={18} color={colors.primary[700]} />
                <Text style={styles.timingTitle}>Watch the behavior track</Text>
              </View>
              <View
                style={styles.timingTrack}
                accessibilityRole="progressbar"
                accessibilityValue={{ min: 0, max: 100, now: Math.round(timingProgress) }}
              >
                <View style={[styles.timingFill, { width: `${timingProgress}%` }]} />
              </View>
              <View style={styles.timingCue}>
                <Text style={styles.timingCueText}>
                  {timingExpired
                    ? 'Round ended. Choose a fresh timing attempt.'
                    : elapsedMs < timing.targetAtMs - 700
                      ? 'Approaching the mat…'
                      : elapsedMs < timing.targetAtMs
                        ? 'Almost there…'
                        : elapsedMs < timing.targetAtMs + 350
                          ? `Target: ${timing.targetLabel}`
                          : 'Behavior moved on'}
                </Text>
              </View>
              <Pressable
                accessibilityRole="button"
                disabled={completing || timingExpired}
                style={[styles.markButton, (completing || timingExpired) && styles.disabled]}
                onPress={markTiming}
              >
                {completing ? (
                  <ActivityIndicator color="#ffffff" />
                ) : (
                  <Text style={styles.markButtonText}>Mark now</Text>
                )}
              </Pressable>
              <Text style={styles.timingBoundary}>
                Device timing is a teaching exercise. Milliseconds never determine public rank or
                professional proficiency.
              </Text>
            </View>
          )}

          {completing && !timing && (
            <View style={styles.scoringRow} accessibilityRole="progressbar">
              <ActivityIndicator color={colors.primary[600]} />
              <Text style={styles.scoringText}>Scoring this practice round…</Text>
            </View>
          )}

          {receipt && (
            <View style={styles.receiptCard}>
              <View style={styles.receiptHeader}>
                <View style={styles.receiptIcon}>
                  <Ionicons
                    name={receipt.correct ? 'checkmark' : 'bulb-outline'}
                    size={20}
                    color={colors.primary[700]}
                  />
                </View>
                <View style={styles.receiptCopy}>
                  <Text style={styles.receiptEyebrow}>PRACTICE RESULT</Text>
                  <Text style={styles.receiptScore}>{receipt.score}/100</Text>
                  <Text style={styles.receiptLabel}>
                    {receipt.correct ? 'Good read' : 'Useful miss — review the pattern'}
                  </Text>
                </View>
              </View>

              {receipt.timingErrorMs !== undefined && (
                <Text style={styles.timingResult}>
                  {receipt.timingErrorMs} ms from the practice target
                </Text>
              )}
              <Text style={styles.receiptExplanation}>{receipt.explanation}</Text>

              <View style={styles.receiptActions}>
                <Pressable
                  accessibilityRole="button"
                  style={styles.playButton}
                  onPress={resetRound}
                >
                  <Ionicons name="game-controller-outline" size={17} color="#ffffff" />
                  <Text style={styles.playButtonText}>Another room</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  accessibilityLabel="Share this human skill moment publicly"
                  disabled={shareState === 'pending' || shareState === 'shared'}
                  style={[
                    styles.shareButton,
                    (shareState === 'pending' || shareState === 'shared') && styles.disabled,
                  ]}
                  onPress={() => void shareResult()}
                >
                  {shareState === 'pending' ? (
                    <ActivityIndicator color={colors.primary[700]} />
                  ) : (
                    <>
                      <Ionicons
                        name={shareState === 'shared' ? 'checkmark-circle' : 'share-social-outline'}
                        size={17}
                        color={colors.primary[700]}
                      />
                      <Text style={styles.shareButtonText}>
                        {shareState === 'shared' ? 'Shared publicly' : 'Share publicly'}
                      </Text>
                    </>
                  )}
                </Pressable>
              </View>

              {shareState === 'error' && (
                <Text style={styles.shareError} accessibilityRole="alert">
                  That result was not shared. Your practice receipt is still saved.
                </Text>
              )}
              <Text style={styles.shareBoundary}>
                Sharing is optional and publishes this human practice moment only. It does not post
                a pet score or private dog history, and reactions do not increase your rank.
              </Text>
            </View>
          )}

          {!receipt && !completing && (
            <Pressable accessibilityRole="button" style={styles.leaveButton} onPress={resetRound}>
              <Text style={styles.leaveButtonText}>
                {timingExpired ? 'Choose a fresh round' : 'Leave round'}
              </Text>
            </Pressable>
          )}
        </View>
      )}

      <View style={styles.safetyCard}>
        <Ionicons name="shield-checkmark-outline" size={21} color={colors.primary[700]} />
        <View style={styles.safetyCopy}>
          <Text style={styles.safetyTitle}>A game is not training authority.</Text>
          <Text style={styles.safetyText}>
            Skillcraft teaches general reward-based mechanics. Significant fear, aggression, pain,
            or sudden behavior change should not become a DIY exposure level; use qualified
            professional or veterinary help when appropriate.
          </Text>
        </View>
      </View>

      <Pressable
        accessibilityRole="button"
        style={styles.backLink}
        onPress={() => navigation.goBack()}
      >
        <Ionicons name="arrow-back" size={16} color={colors.gray[600]} />
        <Text style={styles.backLinkText}>Back to Woof</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 44 },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.paper,
    padding: 24,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  hero: {
    padding: 20,
    borderRadius: 26,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[200],
  },
  heroIcon: {
    width: 48,
    height: 48,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
    marginBottom: 16,
  },
  eyebrow: {
    color: colors.primary[700],
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.4,
  },
  title: { marginTop: 3, color: colors.text.primary, fontSize: 34, fontWeight: '800' },
  subtitle: { marginTop: 8, color: colors.gray[700], fontSize: 14, lineHeight: 21 },
  weekPanel: {
    marginTop: 18,
    padding: 15,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  weekHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 12,
  },
  weekTitle: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  weekSubtitle: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  weekCount: { color: colors.primary[700], fontSize: 18, fontWeight: '900' },
  progressTrack: {
    marginTop: 12,
    height: 9,
    borderRadius: 999,
    overflow: 'hidden',
    backgroundColor: colors.gray[100],
  },
  progressFill: { height: '100%', borderRadius: 999, backgroundColor: colors.primary[500] },
  weekBoundary: { marginTop: 10, color: colors.text.secondary, fontSize: 10, lineHeight: 15 },
  errorCard: {
    marginTop: 14,
    padding: 13,
    borderRadius: 14,
    flexDirection: 'row',
    gap: 8,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 18 },
  challengeList: { marginTop: 18, gap: 12 },
  challengeCard: {
    padding: 17,
    borderRadius: 21,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  challengeHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  challengeIcon: {
    width: 43,
    height: 43,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
  },
  challengeCopy: { flex: 1 },
  roomLabel: {
    color: colors.primary[700],
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  challengeTitle: { marginTop: 2, color: colors.text.primary, fontSize: 17, fontWeight: '800' },
  skillLabel: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: colors.primary[50],
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  badgePracticed: { backgroundColor: colors.success.light },
  badgeText: { color: colors.primary[700], fontSize: 9, fontWeight: '800' },
  badgeTextPracticed: { color: colors.success.dark },
  challengePrompt: { marginTop: 13, color: colors.gray[700], fontSize: 13, lineHeight: 19 },
  personalBest: { marginTop: 8, color: colors.text.secondary, fontSize: 10, fontWeight: '700' },
  playButton: {
    minHeight: 46,
    marginTop: 14,
    paddingHorizontal: 15,
    borderRadius: 13,
    backgroundColor: colors.primary[600],
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
  },
  playButtonText: { color: '#ffffff', fontSize: 13, fontWeight: '800' },
  disabled: { opacity: 0.5 },
  scoringCopy: { color: colors.text.secondary, fontSize: 10, lineHeight: 16, paddingHorizontal: 4 },
  roundCard: {
    marginTop: 18,
    padding: 18,
    borderRadius: 23,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  roundHeader: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  roundIcon: {
    width: 45,
    height: 45,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
  },
  roundCopy: { flex: 1 },
  roundEyebrow: {
    color: colors.primary[700],
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1,
  },
  roundTitle: { marginTop: 2, color: colors.text.primary, fontSize: 20, fontWeight: '800' },
  roundPrompt: { marginTop: 17, color: colors.gray[800], fontSize: 15, lineHeight: 22 },
  optionsList: { marginTop: 16, gap: 9 },
  optionButton: {
    minHeight: 58,
    padding: 14,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: colors.gray[50],
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  optionText: { flex: 1, color: colors.gray[800], fontSize: 13, lineHeight: 19, fontWeight: '600' },
  timingPanel: {
    marginTop: 17,
    padding: 15,
    borderRadius: 18,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  timingTitleRow: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  timingTitle: { color: colors.text.primary, fontSize: 13, fontWeight: '800' },
  timingTrack: {
    marginTop: 14,
    height: 11,
    borderRadius: 999,
    overflow: 'hidden',
    backgroundColor: '#ffffff',
  },
  timingFill: { height: '100%', borderRadius: 999, backgroundColor: colors.primary[500] },
  timingCue: {
    minHeight: 54,
    marginTop: 12,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: colors.primary[100],
    backgroundColor: '#ffffff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
  },
  timingCueText: { color: colors.gray[700], fontSize: 12, lineHeight: 17, textAlign: 'center' },
  markButton: {
    minHeight: 52,
    marginTop: 12,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
  },
  markButtonText: { color: '#ffffff', fontSize: 14, fontWeight: '900' },
  timingBoundary: { marginTop: 10, color: colors.text.secondary, fontSize: 10, lineHeight: 15 },
  scoringRow: {
    marginTop: 18,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  scoringText: { color: colors.text.secondary, fontSize: 12 },
  receiptCard: {
    marginTop: 18,
    padding: 16,
    borderRadius: 19,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  receiptHeader: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  receiptIcon: {
    width: 43,
    height: 43,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
  },
  receiptCopy: { flex: 1 },
  receiptEyebrow: {
    color: colors.primary[700],
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  receiptScore: { marginTop: 1, color: colors.primary[800], fontSize: 29, fontWeight: '900' },
  receiptLabel: { color: colors.text.secondary, fontSize: 10, fontWeight: '700' },
  timingResult: { marginTop: 12, color: colors.primary[800], fontSize: 11, fontWeight: '800' },
  receiptExplanation: { marginTop: 12, color: colors.gray[700], fontSize: 13, lineHeight: 19 },
  receiptActions: { marginTop: 2 },
  shareButton: {
    minHeight: 44,
    marginTop: 8,
    paddingHorizontal: 13,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: colors.primary[200],
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 7,
    backgroundColor: '#ffffff',
  },
  shareButtonText: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  shareError: { marginTop: 9, color: colors.error.dark, fontSize: 10, lineHeight: 15 },
  shareBoundary: { marginTop: 10, color: colors.text.secondary, fontSize: 10, lineHeight: 15 },
  leaveButton: { alignSelf: 'center', paddingVertical: 13, paddingHorizontal: 20, marginTop: 10 },
  leaveButtonText: { color: colors.gray[600], fontSize: 12, fontWeight: '700' },
  safetyCard: {
    marginTop: 18,
    padding: 16,
    borderRadius: 19,
    flexDirection: 'row',
    gap: 10,
    backgroundColor: colors.gray[100],
  },
  safetyCopy: { flex: 1 },
  safetyTitle: { color: colors.text.primary, fontSize: 13, fontWeight: '800' },
  safetyText: { marginTop: 4, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
  backLink: {
    alignSelf: 'center',
    marginTop: 14,
    paddingVertical: 12,
    paddingHorizontal: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  backLinkText: { color: colors.gray[600], fontSize: 12, fontWeight: '700' },
});
