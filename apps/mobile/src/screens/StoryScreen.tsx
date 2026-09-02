import React, { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useFocusEffect } from '@react-navigation/native';
import { storyApi, type StoryDashboard, type StoryMoment } from '../api/story';
import { colors } from '../theme/tokens';

function formatDate(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Unknown date';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() === new Date().getFullYear() ? undefined : 'numeric',
  }).format(date);
}

function iconForMoment(moment: StoryMoment): keyof typeof Ionicons.glyphMap {
  if (moment.sourceType === 'MEDIA') return 'camera-outline';
  if (moment.sourceType === 'ACTIVITY') return 'walk-outline';
  if (moment.pathway === 'BOND') return 'heart-outline';
  if (moment.pathway === 'CARE') return 'shield-checkmark-outline';
  return 'paw-outline';
}

export default function StoryScreen() {
  const [dashboard, setDashboard] = useState<StoryDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      setDashboard(await storyApi.get({ limit: 36 }));
      setError(null);
    } catch {
      setError('Story is unavailable right now. Your existing memories remain unchanged.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load]),
  );

  const moments = useMemo(
    () =>
      [...(dashboard?.moments ?? [])].sort(
        (a, b) => new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime(),
      ),
    [dashboard],
  );

  if (loading && !dashboard) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Gathering the moments that matter…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => void load(true)} />}
    >
      <Text style={styles.eyebrow}>RELATIONSHIP MEMORY</Text>
      <Text style={styles.title}>Story</Text>
      <Text style={styles.subtitle}>
        A record of what you have actually lived together, not a feed you need to keep filling.
      </Text>

      {error && (
        <View style={styles.noticeCard}>
          <Ionicons name="cloud-offline-outline" size={20} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
        </View>
      )}

      {dashboard && (
        <>
          <View style={styles.statsCard}>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{dashboard.stats.activities}</Text>
              <Text style={styles.statLabel}>Activities</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{dashboard.stats.memories}</Text>
              <Text style={styles.statLabel}>Memories</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{dashboard.stats.namedPlaces}</Text>
              <Text style={styles.statLabel}>Places</Text>
            </View>
          </View>

          {dashboard.milestones.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Milestones</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.milestonesRow}>
                {dashboard.milestones.slice(0, 8).map((milestone) => (
                  <View key={milestone.id} style={styles.milestoneCard}>
                    <View style={styles.milestoneIcon}>
                      <Ionicons name="sparkles-outline" size={20} color={colors.primary[700]} />
                    </View>
                    <Text style={styles.milestoneTitle}>{milestone.title}</Text>
                    <Text style={styles.milestoneDescription}>{milestone.description}</Text>
                    <Text style={styles.milestoneDate}>{formatDate(milestone.achievedAt)}</Text>
                  </View>
                ))}
              </ScrollView>
            </View>
          )}

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>What you have lived</Text>
            <Text style={styles.sectionSubtitle}>Recent moments from activity, care, and memories.</Text>

            {moments.length === 0 ? (
              <View style={styles.emptyCard}>
                <Ionicons name="paw-outline" size={28} color={colors.primary[600]} />
                <Text style={styles.emptyTitle}>Your story is just beginning.</Text>
                <Text style={styles.emptyText}>
                  Complete a shared activity or add a memory. Woof will keep the useful parts without turning everyday life into homework.
                </Text>
              </View>
            ) : (
              moments.map((moment) => (
                <View key={moment.id} style={styles.momentCard}>
                  <View style={styles.momentIcon}>
                    <Ionicons name={iconForMoment(moment)} size={20} color={colors.primary[700]} />
                  </View>
                  <View style={styles.momentCopy}>
                    <View style={styles.momentHeader}>
                      <Text style={styles.momentTitle}>{moment.title}</Text>
                      <Text style={styles.momentDate}>{formatDate(moment.occurredAt)}</Text>
                    </View>
                    <Text style={styles.momentSummary}>{moment.summary}</Text>
                    {moment.petNames.length > 0 && (
                      <Text style={styles.petNames}>{moment.petNames.join(' · ')}</Text>
                    )}
                    {moment.curation.note && (
                      <View style={styles.noteCard}>
                        <Ionicons name="bookmark-outline" size={15} color={colors.primary[700]} />
                        <Text style={styles.noteText}>{moment.curation.note}</Text>
                      </View>
                    )}
                  </View>
                </View>
              ))
            )}
          </View>

          <Text style={styles.coverageNote}>
            Story coverage: {dashboard.stats.coverage.toLowerCase()}. Woof may intentionally show a bounded recent history rather than pretending this is every moment you have shared.
          </Text>
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 110 },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.5 },
  title: { marginTop: 3, color: colors.text.primary, fontSize: 34, fontWeight: '800' },
  subtitle: { marginTop: 8, color: colors.text.secondary, fontSize: 15, lineHeight: 22 },
  noticeCard: {
    marginTop: 18,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
    flexDirection: 'row',
    gap: 10,
  },
  noticeText: { flex: 1, color: colors.text.secondary, fontSize: 14, lineHeight: 20 },
  statsCard: {
    marginTop: 20,
    paddingVertical: 18,
    paddingHorizontal: 10,
    borderRadius: 22,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
    flexDirection: 'row',
  },
  stat: { flex: 1, alignItems: 'center' },
  statValue: { color: colors.text.primary, fontSize: 25, fontWeight: '800' },
  statLabel: { marginTop: 4, color: colors.text.secondary, fontSize: 11 },
  section: { marginTop: 26 },
  sectionTitle: { color: colors.text.primary, fontSize: 20, fontWeight: '800' },
  sectionSubtitle: { marginTop: 4, color: colors.text.secondary, fontSize: 13 },
  milestonesRow: { gap: 10, paddingTop: 12, paddingRight: 18 },
  milestoneCard: {
    width: 220,
    padding: 16,
    borderRadius: 18,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  milestoneIcon: {
    width: 38,
    height: 38,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
  },
  milestoneTitle: { marginTop: 10, color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  milestoneDescription: { marginTop: 4, color: colors.text.secondary, fontSize: 12, lineHeight: 17 },
  milestoneDate: { marginTop: 10, color: colors.primary[700], fontSize: 11, fontWeight: '700' },
  emptyCard: {
    marginTop: 12,
    padding: 22,
    borderRadius: 20,
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  emptyTitle: { marginTop: 10, color: colors.text.primary, fontSize: 17, fontWeight: '800' },
  emptyText: { marginTop: 6, color: colors.text.secondary, fontSize: 13, lineHeight: 19, textAlign: 'center' },
  momentCard: {
    marginTop: 11,
    padding: 15,
    borderRadius: 18,
    flexDirection: 'row',
    gap: 11,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  momentIcon: {
    width: 40,
    height: 40,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
  },
  momentCopy: { flex: 1 },
  momentHeader: { flexDirection: 'row', alignItems: 'flex-start', gap: 8 },
  momentTitle: { flex: 1, color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  momentDate: { color: colors.text.secondary, fontSize: 10 },
  momentSummary: { marginTop: 5, color: colors.gray[700], fontSize: 13, lineHeight: 19 },
  petNames: { marginTop: 7, color: colors.primary[700], fontSize: 11, fontWeight: '700' },
  noteCard: {
    marginTop: 9,
    padding: 10,
    borderRadius: 12,
    flexDirection: 'row',
    gap: 7,
    backgroundColor: colors.primary[50],
  },
  noteText: { flex: 1, color: colors.gray[700], fontSize: 12, lineHeight: 17 },
  coverageNote: { marginTop: 22, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
});
