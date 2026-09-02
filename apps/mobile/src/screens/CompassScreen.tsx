import React, { useCallback, useState } from 'react';
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
import { adventureApi, type AdventureDashboard, type CompassPathway } from '../api/adventure';
import { colors } from '../theme/tokens';

const pathwayIcon: Record<string, keyof typeof Ionicons.glyphMap> = {
  MOVE: 'walk-outline',
  EXPLORE: 'compass-outline',
  ENRICH: 'sparkles-outline',
  LEARN: 'school-outline',
  CONNECT: 'people-outline',
  CARE: 'shield-checkmark-outline',
  RECOVER: 'moon-outline',
  BOND: 'heart-outline',
};

function PathwayCard({ item }: { item: CompassPathway }) {
  const coverage = Math.max(0, Math.min(1, item.coverage));
  return (
    <View style={styles.pathwayCard}>
      <View style={styles.pathwayHeader}>
        <View style={styles.pathwayIcon}>
          <Ionicons name={pathwayIcon[item.pathway] ?? 'paw-outline'} size={20} color={colors.primary[700]} />
        </View>
        <View style={styles.pathwayCopy}>
          <Text style={styles.pathwayLabel}>{item.label}</Text>
          <Text style={styles.pathwayMeta}>
            {item.recentDays} recent {item.recentDays === 1 ? 'day' : 'days'} · {item.xp} XP
          </Text>
        </View>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${coverage * 100}%` }]} />
      </View>
      <Text style={styles.coverageText}>
        {Math.round(coverage * 100)}% recent pathway coverage
      </Text>
    </View>
  );
}

export default function CompassScreen() {
  const [dashboard, setDashboard] = useState<AdventureDashboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      setDashboard(await adventureApi.getMine());
      setError(null);
    } catch {
      setError('Compass is unavailable right now. Woof has not changed any relationship evidence.');
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

  if (loading && !dashboard) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Reading your recent rhythm…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => void load(true)} />}
    >
      <Text style={styles.eyebrow}>RELATIONSHIP CONTEXT</Text>
      <Text style={styles.title}>Compass</Text>
      <Text style={styles.subtitle}>
        What you have been exploring together. This is context for better choices, not a score for your dog.
      </Text>

      {error && (
        <View style={styles.noticeCard}>
          <Ionicons name="cloud-offline-outline" size={20} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
        </View>
      )}

      {dashboard && (
        <>
          <View style={styles.summaryCard}>
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>{dashboard.bondXp}</Text>
              <Text style={styles.summaryLabel}>Bond XP</Text>
            </View>
            <View style={styles.divider} />
            <View style={styles.summaryItem}>
              <Text style={styles.summaryValue}>{dashboard.rhythm.activeWeeks}/{dashboard.rhythm.windowWeeks}</Text>
              <Text style={styles.summaryLabel}>Active weeks</Text>
            </View>
          </View>
          <Text style={styles.rhythmCopy}>{dashboard.rhythm.label}</Text>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>{dashboard.pet.name}&apos;s recent shape</Text>
            <Text style={styles.sectionSubtitle}>Different dogs should form different shapes.</Text>
          </View>

          {dashboard.compass.map((item) => (
            <PathwayCard key={item.pathway} item={item} />
          ))}

          {dashboard.learningSummary.length > 0 && (
            <View style={styles.learningCard}>
              <Text style={styles.eyebrow}>CURRENT LEARNING</Text>
              {dashboard.learningSummary.slice(0, 4).map((line) => (
                <View key={line} style={styles.learningRow}>
                  <Ionicons name="sparkles-outline" size={16} color={colors.primary[600]} />
                  <Text style={styles.learningText}>{line}</Text>
                </View>
              ))}
            </View>
          )}

          <Text style={styles.disclaimer}>{dashboard.disclaimer}</Text>
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
    backgroundColor: colors.background.paper,
    padding: 24,
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
  summaryCard: {
    marginTop: 20,
    padding: 18,
    borderRadius: 22,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[200],
    flexDirection: 'row',
    alignItems: 'center',
  },
  summaryItem: { flex: 1, alignItems: 'center' },
  summaryValue: { color: colors.text.primary, fontSize: 26, fontWeight: '800' },
  summaryLabel: { marginTop: 4, color: colors.text.secondary, fontSize: 12 },
  divider: { width: 1, height: 42, backgroundColor: colors.gray[200] },
  rhythmCopy: { marginTop: 10, color: colors.text.secondary, fontSize: 13, textAlign: 'center' },
  sectionHeader: { marginTop: 28, marginBottom: 4 },
  sectionTitle: { color: colors.text.primary, fontSize: 20, fontWeight: '800' },
  sectionSubtitle: { marginTop: 4, color: colors.text.secondary, fontSize: 13 },
  pathwayCard: {
    marginTop: 10,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  pathwayHeader: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  pathwayIcon: {
    width: 40,
    height: 40,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
  },
  pathwayCopy: { flex: 1 },
  pathwayLabel: { color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  pathwayMeta: { marginTop: 2, color: colors.text.secondary, fontSize: 12 },
  track: { marginTop: 13, height: 7, borderRadius: 999, backgroundColor: colors.gray[100], overflow: 'hidden' },
  fill: { height: '100%', borderRadius: 999, backgroundColor: colors.primary[500] },
  coverageText: { marginTop: 7, color: colors.text.secondary, fontSize: 11 },
  learningCard: { marginTop: 24, padding: 18, borderRadius: 20, backgroundColor: colors.primary[50] },
  learningRow: { marginTop: 10, flexDirection: 'row', gap: 8, alignItems: 'flex-start' },
  learningText: { flex: 1, color: colors.gray[700], fontSize: 14, lineHeight: 20 },
  disclaimer: { marginTop: 22, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
});
