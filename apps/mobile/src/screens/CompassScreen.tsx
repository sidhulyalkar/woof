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
import { deriveAdventureTrail, TRAIL_PATHWAYS, type TrailPathway } from '../game/adventure-trail';
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

const pathwayShortLabel: Record<TrailPathway, string> = {
  MOVE: 'Move',
  EXPLORE: 'Explore',
  ENRICH: 'Enrich',
  LEARN: 'Learn',
  CONNECT: 'Connect',
  RECOVER: 'Recover',
  BOND: 'Bond',
};

function PathwayCard({ item }: { item: CompassPathway }) {
  const coverage = Math.max(0, Math.min(1, item.coverage));
  return (
    <View style={styles.pathwayCard}>
      <View style={styles.pathwayHeader}>
        <View style={styles.pathwayIcon}>
          <Ionicons
            name={pathwayIcon[item.pathway] ?? 'paw-outline'}
            size={20}
            color={colors.primary[700]}
          />
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
      <Text style={styles.coverageText}>{Math.round(coverage * 100)}% recent pathway coverage</Text>
    </View>
  );
}

function DiscoveryStamp({ pathway, discovered }: { pathway: TrailPathway; discovered: boolean }) {
  return (
    <View style={[styles.stamp, !discovered && styles.stampUndiscovered]}>
      <View style={[styles.stampIcon, !discovered && styles.stampIconUndiscovered]}>
        <Ionicons
          name={discovered ? (pathwayIcon[pathway] ?? 'paw-outline') : 'lock-closed-outline'}
          size={17}
          color={discovered ? colors.primary[700] : colors.gray[500]}
        />
      </View>
      <Text style={[styles.stampLabel, !discovered && styles.stampLabelUndiscovered]}>
        {pathwayShortLabel[pathway]}
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
    }, [load])
  );

  const trail = dashboard ? deriveAdventureTrail(dashboard) : null;

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
        What you have been exploring together. This is context for better choices, not a score for
        your dog.
      </Text>

      {error && (
        <View style={styles.noticeCard}>
          <Ionicons name="cloud-offline-outline" size={20} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
        </View>
      )}

      {dashboard && trail && (
        <View style={styles.trailCard}>
          <View style={styles.trailHeader}>
            <View style={styles.trailIcon}>
              <Ionicons name="map-outline" size={22} color={colors.primary[700]} />
            </View>
            <View style={styles.trailHeaderCopy}>
              <Text style={styles.eyebrow}>ADVENTURE TRAIL</Text>
              <Text style={styles.trailTitle}>{trail.chapter.label}</Text>
            </View>
          </View>

          <Text style={styles.trailDescription}>{trail.chapter.description}</Text>

          <View style={styles.trailTrack}>
            <View
              style={[styles.trailFill, { width: `${Math.round(trail.chapterProgress * 100)}%` }]}
            />
          </View>
          <View style={styles.trailProgressRow}>
            <Text style={styles.trailProgressValue}>{trail.trailXp} Trail XP</Text>
            <Text style={styles.trailProgressHint}>
              {trail.nextChapter
                ? `${trail.xpToNextChapter} to ${trail.nextChapter.label}`
                : 'The trail keeps unfolding'}
            </Text>
          </View>
          <Text style={styles.trailAuthorityCopy}>
            Trail XP is a display-only sum of server-earned XP from seven non-care pathways. CARE
            never advances chapters or changes recommendations.
          </Text>

          <View style={styles.discoveryHeader}>
            <View>
              <Text style={styles.discoveryTitle}>Discovery stamps</Text>
              <Text style={styles.discoverySubtitle}>
                Different kinds of good days leave a mark.
              </Text>
            </View>
            <Text style={styles.discoveryCount}>
              {trail.discoveryCount}/{trail.discoveryTotal}
            </Text>
          </View>
          <View style={styles.stampsRow}>
            {TRAIL_PATHWAYS.map((pathway) => (
              <DiscoveryStamp
                key={pathway}
                pathway={pathway}
                discovered={trail.discoveredPathways.includes(pathway)}
              />
            ))}
          </View>
          <Text style={styles.collectionBoundary}>
            CARE stays visible in the Compass below, but it is intentionally outside this collection
            game.
          </Text>

          <View style={styles.rhythmPanel}>
            <View style={styles.rhythmHeader}>
              <View>
                <Text style={styles.rhythmTitle}>Rolling Rhythm</Text>
                <Text style={styles.rhythmSubtitle}>Meaningful weeks, not perfect days.</Text>
              </View>
              <Text style={styles.rhythmValue}>{trail.activeWeeks}/{trail.windowWeeks}</Text>
            </View>
            <View style={styles.rhythmSlots}>
              {Array.from({ length: trail.windowWeeks }, (_, index) => (
                <View
                  key={index}
                  style={[styles.rhythmSlot, index < trail.activeWeeks && styles.rhythmSlotActive]}
                >
                  <Ionicons
                    name={index < trail.activeWeeks ? 'paw' : 'paw-outline'}
                    size={16}
                    color={index < trail.activeWeeks ? colors.primary[700] : colors.gray[400]}
                  />
                </View>
              ))}
            </View>
            <Text style={styles.rhythmBoundary}>
              Missing a day never resets Rhythm. Recovery and listening can be real progress too.
            </Text>
          </View>
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
              <Text style={styles.summaryValue}>
                {dashboard.rhythm.activeWeeks}/{dashboard.rhythm.windowWeeks}
              </Text>
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
  trailCard: {
    marginTop: 20,
    padding: 18,
    borderRadius: 24,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[200],
  },
  trailHeader: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  trailIcon: {
    width: 44,
    height: 44,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  trailHeaderCopy: { flex: 1 },
  trailTitle: { marginTop: 2, color: colors.text.primary, fontSize: 21, fontWeight: '800' },
  trailDescription: {
    marginTop: 12,
    color: colors.gray[700],
    fontSize: 14,
    lineHeight: 20,
  },
  trailTrack: {
    marginTop: 16,
    height: 9,
    borderRadius: 999,
    backgroundColor: '#ffffff',
    overflow: 'hidden',
  },
  trailFill: { height: '100%', borderRadius: 999, backgroundColor: colors.primary[500] },
  trailProgressRow: {
    marginTop: 7,
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  trailProgressValue: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  trailProgressHint: {
    flex: 1,
    color: colors.text.secondary,
    fontSize: 11,
    textAlign: 'right',
  },
  trailAuthorityCopy: {
    marginTop: 10,
    color: colors.text.secondary,
    fontSize: 11,
    lineHeight: 16,
  },
  discoveryHeader: {
    marginTop: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    gap: 12,
  },
  discoveryTitle: { color: colors.text.primary, fontSize: 15, fontWeight: '800' },
  discoverySubtitle: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  discoveryCount: { color: colors.primary[700], fontSize: 16, fontWeight: '800' },
  stampsRow: { marginTop: 11, flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  stamp: {
    width: 72,
    paddingVertical: 10,
    paddingHorizontal: 6,
    borderRadius: 14,
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  stampUndiscovered: { backgroundColor: colors.gray[50], borderColor: colors.gray[200] },
  stampIcon: {
    width: 31,
    height: 31,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[50],
  },
  stampIconUndiscovered: { backgroundColor: colors.gray[100] },
  stampLabel: { marginTop: 6, color: colors.text.primary, fontSize: 10, fontWeight: '700' },
  stampLabelUndiscovered: { color: colors.text.secondary },
  collectionBoundary: {
    marginTop: 9,
    color: colors.text.secondary,
    fontSize: 10,
    lineHeight: 15,
  },
  rhythmPanel: {
    marginTop: 18,
    padding: 14,
    borderRadius: 17,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  rhythmHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 12,
  },
  rhythmTitle: { color: colors.text.primary, fontSize: 14, fontWeight: '800' },
  rhythmSubtitle: { marginTop: 2, color: colors.text.secondary, fontSize: 11 },
  rhythmValue: { color: colors.primary[700], fontSize: 16, fontWeight: '800' },
  rhythmSlots: { marginTop: 11, flexDirection: 'row', gap: 7 },
  rhythmSlot: {
    width: 34,
    height: 34,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.gray[50],
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  rhythmSlotActive: {
    backgroundColor: colors.primary[50],
    borderColor: colors.primary[200],
  },
  rhythmBoundary: {
    marginTop: 10,
    color: colors.text.secondary,
    fontSize: 10,
    lineHeight: 15,
  },
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
  track: {
    marginTop: 13,
    height: 7,
    borderRadius: 999,
    backgroundColor: colors.gray[100],
    overflow: 'hidden',
  },
  fill: { height: '100%', borderRadius: 999, backgroundColor: colors.primary[500] },
  coverageText: { marginTop: 7, color: colors.text.secondary, fontSize: 11 },
  learningCard: {
    marginTop: 24,
    padding: 18,
    borderRadius: 20,
    backgroundColor: colors.primary[50],
  },
  learningRow: { marginTop: 10, flexDirection: 'row', gap: 8, alignItems: 'flex-start' },
  learningText: { flex: 1, color: colors.gray[700], fontSize: 14, lineHeight: 20 },
  disclaimer: { marginTop: 22, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
});
