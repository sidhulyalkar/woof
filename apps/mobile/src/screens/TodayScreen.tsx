import React, { useCallback, useState } from 'react';
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
import { CompositeScreenProps, useFocusEffect } from '@react-navigation/native';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';
import type { StackScreenProps } from '@react-navigation/stack';
import { adventureApi, type AdventureDashboard, type AdventureQuest } from '../api/adventure';
import { colors } from '../theme/tokens';
import type { MainTabParamList, RootStackParamList } from '../navigation/AppNavigator';

type Props = CompositeScreenProps<
  BottomTabScreenProps<MainTabParamList, 'Today'>,
  StackScreenProps<RootStackParamList>
>;

type DogExperience = 'loved_it' | 'comfortable' | 'not_their_thing';
type OwnerExperience = 'great' | 'fine' | 'a_lot_today';

const dogChoices: {
  value: DogExperience;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  { value: 'loved_it', label: 'Loved it', icon: 'happy-outline' },
  { value: 'comfortable', label: 'Comfortable', icon: 'checkmark-circle-outline' },
  { value: 'not_their_thing', label: 'Not their thing', icon: 'remove-circle-outline' },
];

const ownerChoices: {
  value: OwnerExperience;
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  { value: 'great', label: 'Great', icon: 'sparkles-outline' },
  { value: 'fine', label: 'Fine', icon: 'thumbs-up-outline' },
  { value: 'a_lot_today', label: 'A lot today', icon: 'battery-half-outline' },
];

const toolLinks: {
  route: keyof RootStackParamList;
  label: string;
  caption: string;
  icon: keyof typeof Ionicons.glyphMap;
}[] = [
  {
    route: 'DailySignals',
    label: 'Daily Signals',
    caption: 'A quick check-in',
    icon: 'pulse-outline',
  },
  { route: 'Pets', label: 'Pets', caption: 'Profiles and context', icon: 'paw-outline' },
  { route: 'Goals', label: 'Goals', caption: 'Shared intentions', icon: 'flag-outline' },
  { route: 'Library', label: 'Library', caption: 'Photos and memories', icon: 'images-outline' },
  { route: 'Events', label: 'Events', caption: 'Community plans', icon: 'calendar-outline' },
  {
    route: 'Profile',
    label: 'You',
    caption: 'Account and settings',
    icon: 'person-circle-outline',
  },
];

export default function TodayScreen({ navigation }: Props) {
  const [dashboard, setDashboard] = useState<AdventureDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeQuestId, setActiveQuestId] = useState<string | null>(null);
  const [closingQuest, setClosingQuest] = useState<AdventureQuest | null>(null);
  const [dogExperience, setDogExperience] = useState<DogExperience | null>(null);
  const [ownerExperience, setOwnerExperience] = useState<OwnerExperience | null>(null);
  const [safeOptOut, setSafeOptOut] = useState(false);
  const [savingOutcome, setSavingOutcome] = useState(false);
  const [receipt, setReceipt] = useState<string | null>(null);

  const load = useCallback(async (asRefresh = false) => {
    if (asRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const next = await adventureApi.getMine();
      setDashboard(next);
      setError(null);
    } catch {
      setError(
        'Woof could not load a recommendation right now. Your existing relationship data is unchanged.'
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const startQuest = async (quest: AdventureQuest) => {
    if (!dashboard) return;
    setActiveQuestId(quest.id);
    setReceipt(null);
    try {
      await adventureApi.selectQuest(quest.id, dashboard.pet.id);
    } catch {
      // Selection persistence improves continuity but must not block the real-world action.
    }
  };

  const openOutcome = (quest: AdventureQuest, optOut = false) => {
    setClosingQuest(quest);
    setDogExperience(null);
    setOwnerExperience(null);
    setSafeOptOut(optOut);
    setReceipt(null);
  };

  const closeOutcome = () => {
    setClosingQuest(null);
    setDogExperience(null);
    setOwnerExperience(null);
    setSafeOptOut(false);
  };

  const saveOutcome = async () => {
    if (!dashboard || !closingQuest || !dogExperience || !ownerExperience) return;
    setSavingOutcome(true);
    try {
      const result = await adventureApi.completeQuest(closingQuest.id, {
        petId: dashboard.pet.id,
        dogExperience,
        ownerExperience,
        safeOptOut,
      });
      const reward = result.reward.duplicate
        ? 'This outcome was already saved.'
        : result.reward.bondXp > 0
          ? ` +${result.reward.bondXp} Bond XP.`
          : '';
      setReceipt(`${result.message}${reward}`);
      setActiveQuestId(null);
      closeOutcome();
      await load(true);
    } catch {
      setReceipt('Woof could not save that outcome yet. You can try closing the loop again.');
    } finally {
      setSavingOutcome(false);
    }
  };

  const primaryQuest = dashboard?.quests[0] ?? null;
  const alternatives = dashboard?.quests.slice(1, 3) ?? [];

  if (loading && !dashboard) {
    return (
      <View style={styles.centered} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Finding one good thing to do together…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => void load(true)} />}
    >
      <View style={styles.heroRow}>
        <View style={styles.brandMark}>
          <Ionicons name="paw" size={22} color="#ffffff" />
        </View>
        <View style={styles.heroCopy}>
          <Text style={styles.eyebrow}>DOG + HUMAN</Text>
          <Text style={styles.title}>Today</Text>
        </View>
      </View>

      {dashboard && (
        <Text style={styles.intro}>
          One useful next step with {dashboard.pet.name}. Woof recommends, you choose.
        </Text>
      )}

      {error && (
        <View style={styles.noticeCard}>
          <Ionicons name="cloud-offline-outline" size={20} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
          <Pressable
            accessibilityRole="button"
            style={styles.secondaryButton}
            onPress={() => void load()}
          >
            <Text style={styles.secondaryButtonText}>Try again</Text>
          </Pressable>
        </View>
      )}

      {receipt && (
        <View style={styles.receiptCard} accessibilityRole="summary">
          <Ionicons name="sparkles-outline" size={20} color={colors.primary[700]} />
          <Text style={styles.receiptText}>{receipt}</Text>
        </View>
      )}

      {dashboard && primaryQuest && (
        <View style={styles.primaryCard}>
          <View style={styles.questHeader}>
            <View style={styles.questIcon}>
              <Ionicons name="compass-outline" size={22} color={colors.primary[700]} />
            </View>
            <View style={styles.questHeaderCopy}>
              <Text style={styles.eyebrow}>
                A GOOD PLACE TO START WITH {dashboard.pet.name.toUpperCase()}
              </Text>
              <Text style={styles.questTitle}>{primaryQuest.title}</Text>
            </View>
          </View>
          <Text style={styles.questDescription}>{primaryQuest.description}</Text>
          <View style={styles.whyCard}>
            <Text style={styles.whyLabel}>WHY THIS ONE TODAY</Text>
            <Text style={styles.whyText}>{primaryQuest.why}</Text>
          </View>

          {activeQuestId === primaryQuest.id ? (
            <View style={styles.startedCard}>
              <Ionicons name="walk-outline" size={22} color={colors.success.dark} />
              <View style={styles.startedCopy}>
                <Text style={styles.startedTitle}>Go be together.</Text>
                <Text style={styles.startedText}>
                  The phone can wait. Come back when you want to close the loop.
                </Text>
              </View>
            </View>
          ) : (
            <Pressable
              accessibilityRole="button"
              style={styles.primaryButton}
              onPress={() => void startQuest(primaryQuest)}
            >
              <Text style={styles.primaryButtonText}>
                {primaryQuest.actionLabel || 'Start together'}
              </Text>
            </Pressable>
          )}

          <View style={styles.actionRow}>
            <Pressable
              accessibilityRole="button"
              style={styles.outlineButton}
              onPress={() => openOutcome(primaryQuest)}
            >
              <Ionicons name="checkmark-circle-outline" size={18} color={colors.gray[800]} />
              <Text style={styles.outlineButtonText}>Close the loop</Text>
            </Pressable>
            {primaryQuest.safeStopEligible && (
              <Pressable
                accessibilityRole="button"
                style={styles.ghostButton}
                onPress={() => openOutcome(primaryQuest, true)}
              >
                <Text style={styles.ghostButtonText}>I listened and stopped</Text>
              </Pressable>
            )}
          </View>
          <Text style={styles.permissionText}>
            Making it easier, changing your mind, or stopping when your dog is done can all be the
            right outcome.
          </Text>
        </View>
      )}

      {dashboard && !primaryQuest && (
        <View style={styles.noticeCard}>
          <Ionicons name="moon-outline" size={22} color={colors.primary[700]} />
          <Text style={styles.noticeTitle}>Nothing needs pushing today.</Text>
          <Text style={styles.noticeText}>Rest or your usual routine is a valid choice.</Text>
        </View>
      )}

      {alternatives.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Different kind of day?</Text>
          {alternatives.map((quest) => (
            <View key={quest.id} style={styles.alternativeCard}>
              <Text style={styles.alternativeKind}>
                {quest.primaryPathway} · {quest.variant}
              </Text>
              <Text style={styles.alternativeTitle}>{quest.title}</Text>
              <Text style={styles.alternativeText}>{quest.description}</Text>
              <View style={styles.actionRow}>
                <Pressable
                  accessibilityRole="button"
                  style={styles.smallButton}
                  onPress={() => void startQuest(quest)}
                >
                  <Text style={styles.smallButtonText}>{quest.actionLabel || 'Start'}</Text>
                </Pressable>
                <Pressable
                  accessibilityRole="button"
                  style={styles.ghostButton}
                  onPress={() => openOutcome(quest)}
                >
                  <Text style={styles.ghostButtonText}>Close loop</Text>
                </Pressable>
              </View>
            </View>
          ))}
        </View>
      )}

      {dashboard && dashboard.learningSummary.length > 0 && (
        <View style={styles.learningCard}>
          <Text style={styles.eyebrow}>WHAT WOOF IS LEARNING</Text>
          {dashboard.learningSummary.slice(0, 3).map((line) => (
            <View key={line} style={styles.learningRow}>
              <Ionicons name="sparkles-outline" size={16} color={colors.primary[600]} />
              <Text style={styles.learningText}>{line}</Text>
            </View>
          ))}
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Relationship tools</Text>
        <Text style={styles.sectionSubtitle}>
          Useful when you need them. They are not extra assignments.
        </Text>
        <View style={styles.toolsGrid}>
          {toolLinks.map((tool) => (
            <Pressable
              key={tool.route}
              accessibilityRole="button"
              style={styles.toolCard}
              onPress={() => navigation.navigate(tool.route as never)}
            >
              <Ionicons name={tool.icon} size={22} color={colors.primary[700]} />
              <Text style={styles.toolTitle}>{tool.label}</Text>
              <Text style={styles.toolCaption}>{tool.caption}</Text>
            </Pressable>
          ))}
        </View>
      </View>

      {closingQuest && (
        <View style={styles.outcomeCard} accessibilityViewIsModal>
          <View style={styles.outcomeHeader}>
            <Text style={styles.outcomeTitle}>How did {closingQuest.title} go?</Text>
            <Pressable accessibilityRole="button" onPress={closeOutcome} hitSlop={12}>
              <Ionicons name="close" size={24} color={colors.gray[700]} />
            </Pressable>
          </View>

          {safeOptOut && (
            <View style={styles.safeStopBanner}>
              <Ionicons name="heart-outline" size={18} color={colors.success.dark} />
              <Text style={styles.safeStopText}>Listening and stopping is a valid outcome.</Text>
            </View>
          )}

          <Text style={styles.questionLabel}>How was it for your dog?</Text>
          <View style={styles.choiceWrap}>
            {dogChoices.map((choice) => (
              <Pressable
                key={choice.value}
                accessibilityRole="button"
                accessibilityState={{ selected: dogExperience === choice.value }}
                style={[styles.choice, dogExperience === choice.value && styles.choiceSelected]}
                onPress={() => setDogExperience(choice.value)}
              >
                <Ionicons name={choice.icon} size={18} color={colors.gray[800]} />
                <Text style={styles.choiceText}>{choice.label}</Text>
              </Pressable>
            ))}
          </View>

          <Text style={styles.questionLabel}>How was it for you?</Text>
          <View style={styles.choiceWrap}>
            {ownerChoices.map((choice) => (
              <Pressable
                key={choice.value}
                accessibilityRole="button"
                accessibilityState={{ selected: ownerExperience === choice.value }}
                style={[styles.choice, ownerExperience === choice.value && styles.choiceSelected]}
                onPress={() => setOwnerExperience(choice.value)}
              >
                <Ionicons name={choice.icon} size={18} color={colors.gray[800]} />
                <Text style={styles.choiceText}>{choice.label}</Text>
              </Pressable>
            ))}
          </View>

          <Pressable
            accessibilityRole="button"
            accessibilityState={{ disabled: !dogExperience || !ownerExperience || savingOutcome }}
            disabled={!dogExperience || !ownerExperience || savingOutcome}
            style={[
              styles.primaryButton,
              (!dogExperience || !ownerExperience || savingOutcome) && styles.buttonDisabled,
            ]}
            onPress={() => void saveOutcome()}
          >
            {savingOutcome ? (
              <ActivityIndicator color="#ffffff" />
            ) : (
              <Text style={styles.primaryButtonText}>Save what Woof should learn</Text>
            )}
          </Pressable>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 120 },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary, fontSize: 14 },
  heroRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  brandMark: {
    width: 44,
    height: 44,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
  },
  heroCopy: { flex: 1 },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.6 },
  title: { marginTop: 2, color: colors.text.primary, fontSize: 34, fontWeight: '800' },
  intro: { marginTop: 12, color: colors.text.secondary, fontSize: 15, lineHeight: 22 },
  noticeCard: {
    marginTop: 18,
    padding: 18,
    borderRadius: 20,
    backgroundColor: colors.background.elevated,
    borderWidth: 1,
    borderColor: colors.gray[200],
    gap: 8,
  },
  noticeTitle: { color: colors.text.primary, fontSize: 18, fontWeight: '700' },
  noticeText: { color: colors.text.secondary, fontSize: 14, lineHeight: 20 },
  receiptCard: {
    marginTop: 18,
    padding: 16,
    borderRadius: 18,
    flexDirection: 'row',
    gap: 10,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[200],
  },
  receiptText: { flex: 1, color: colors.primary[900], fontSize: 14, lineHeight: 20 },
  primaryCard: {
    marginTop: 20,
    padding: 20,
    borderRadius: 26,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[200],
  },
  questHeader: { flexDirection: 'row', gap: 12, alignItems: 'flex-start' },
  questIcon: {
    width: 44,
    height: 44,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[100],
  },
  questHeaderCopy: { flex: 1 },
  questTitle: {
    marginTop: 4,
    color: colors.text.primary,
    fontSize: 25,
    lineHeight: 30,
    fontWeight: '800',
  },
  questDescription: { marginTop: 14, color: colors.gray[800], fontSize: 16, lineHeight: 23 },
  whyCard: { marginTop: 14, padding: 14, borderRadius: 16, backgroundColor: colors.gray[50] },
  whyLabel: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.2 },
  whyText: { marginTop: 5, color: colors.gray[700], fontSize: 14, lineHeight: 20 },
  primaryButton: {
    marginTop: 16,
    minHeight: 50,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
    backgroundColor: colors.primary[600],
  },
  primaryButtonText: { color: '#ffffff', fontSize: 16, fontWeight: '700' },
  secondaryButton: {
    alignSelf: 'flex-start',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 12,
    backgroundColor: colors.gray[100],
  },
  secondaryButtonText: { color: colors.gray[800], fontWeight: '700' },
  startedCard: {
    marginTop: 16,
    padding: 14,
    borderRadius: 16,
    flexDirection: 'row',
    gap: 10,
    backgroundColor: colors.success.light,
  },
  startedCopy: { flex: 1 },
  startedTitle: { color: colors.success.dark, fontWeight: '800', fontSize: 15 },
  startedText: { marginTop: 3, color: colors.success.dark, fontSize: 13, lineHeight: 18 },
  actionRow: { marginTop: 12, flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  outlineButton: {
    minHeight: 42,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 13,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.gray[300],
  },
  outlineButtonText: { color: colors.gray[800], fontSize: 13, fontWeight: '700' },
  ghostButton: { minHeight: 42, justifyContent: 'center', paddingHorizontal: 8 },
  ghostButtonText: { color: colors.primary[700], fontSize: 13, fontWeight: '700' },
  permissionText: { marginTop: 12, color: colors.text.secondary, fontSize: 12, lineHeight: 18 },
  section: { marginTop: 26 },
  sectionTitle: { color: colors.text.primary, fontSize: 19, fontWeight: '800' },
  sectionSubtitle: { marginTop: 4, color: colors.text.secondary, fontSize: 13, lineHeight: 18 },
  alternativeCard: {
    marginTop: 10,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  alternativeKind: {
    color: colors.text.secondary,
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.9,
  },
  alternativeTitle: { marginTop: 4, color: colors.text.primary, fontSize: 17, fontWeight: '800' },
  alternativeText: { marginTop: 6, color: colors.text.secondary, fontSize: 13, lineHeight: 19 },
  smallButton: {
    minHeight: 40,
    justifyContent: 'center',
    paddingHorizontal: 14,
    borderRadius: 12,
    backgroundColor: colors.primary[100],
  },
  smallButtonText: { color: colors.primary[800], fontWeight: '700' },
  learningCard: {
    marginTop: 22,
    padding: 18,
    borderRadius: 20,
    backgroundColor: colors.primary[50],
  },
  learningRow: { marginTop: 10, flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  learningText: { flex: 1, color: colors.gray[700], fontSize: 14, lineHeight: 20 },
  toolsGrid: { marginTop: 12, flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  toolCard: {
    width: '48%',
    minHeight: 112,
    padding: 14,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  toolTitle: { marginTop: 10, color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  toolCaption: { marginTop: 3, color: colors.text.secondary, fontSize: 12, lineHeight: 16 },
  outcomeCard: {
    marginTop: 28,
    padding: 20,
    borderRadius: 24,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[200],
  },
  outcomeHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  outcomeTitle: {
    flex: 1,
    color: colors.text.primary,
    fontSize: 21,
    lineHeight: 27,
    fontWeight: '800',
  },
  safeStopBanner: {
    marginTop: 14,
    padding: 12,
    borderRadius: 14,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: colors.success.light,
  },
  safeStopText: { flex: 1, color: colors.success.dark, fontSize: 13, fontWeight: '600' },
  questionLabel: {
    marginTop: 18,
    marginBottom: 8,
    color: colors.gray[800],
    fontSize: 14,
    fontWeight: '800',
  },
  choiceWrap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  choice: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    borderRadius: 13,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
  },
  choiceSelected: { borderColor: colors.primary[500], backgroundColor: colors.primary[50] },
  choiceText: { color: colors.gray[800], fontSize: 13, fontWeight: '600' },
  buttonDisabled: { opacity: 0.45 },
});
