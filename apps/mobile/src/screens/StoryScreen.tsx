import React, { useCallback, useMemo, useRef, useState } from 'react';
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
import { householdsApi } from '../api/households';
import { storyApi, type StoryDashboard, type StoryMoment } from '../api/story';
import { colors } from '../theme/tokens';

type StoryScope = 'ALL' | string;

type StoryPet = {
  id: string;
  name: string;
};

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

function uniqueStoryPets(
  households: Awaited<ReturnType<typeof householdsApi.getMine>>
): StoryPet[] {
  const pets = new Map<string, StoryPet>();
  for (const household of households) {
    for (const link of household.pets) {
      if (!pets.has(link.pet.id)) pets.set(link.pet.id, { id: link.pet.id, name: link.pet.name });
    }
  }
  return [...pets.values()];
}

export default function StoryScreen() {
  const [dashboard, setDashboard] = useState<StoryDashboard | null>(null);
  const [dashboardScope, setDashboardScope] = useState<StoryScope | null>(null);
  const [filterPets, setFilterPets] = useState<StoryPet[]>([]);
  const [selectedScope, setSelectedScopeState] = useState<StoryScope>('ALL');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filterLoading, setFilterLoading] = useState(true);
  const [filterError, setFilterError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectedScopeRef = useRef<StoryScope>('ALL');
  const storyRequestRef = useRef(0);
  const filterRequestRef = useRef(0);

  const setSelectedScope = useCallback((scope: StoryScope) => {
    selectedScopeRef.current = scope;
    setSelectedScopeState(scope);
  }, []);

  const loadStory = useCallback(async (refresh = false, scope = selectedScopeRef.current) => {
    const requestId = ++storyRequestRef.current;
    if (refresh) setRefreshing(true);
    else setLoading(true);
    try {
      const next = await storyApi.get({ limit: 36, ...(scope === 'ALL' ? {} : { petId: scope }) });
      if (requestId !== storyRequestRef.current || scope !== selectedScopeRef.current) return;
      setDashboard(next);
      setDashboardScope(scope);
      setError(null);
    } catch {
      if (requestId !== storyRequestRef.current || scope !== selectedScopeRef.current) return;
      setError('Story is unavailable right now. Your existing memories remain unchanged.');
    } finally {
      if (requestId === storyRequestRef.current && scope === selectedScopeRef.current) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, []);

  const loadFilters = useCallback(async () => {
    const requestId = ++filterRequestRef.current;
    setFilterLoading(true);
    try {
      const pets = uniqueStoryPets(await householdsApi.getMine());
      if (requestId !== filterRequestRef.current) return;
      setFilterPets(pets);
      setFilterError(null);
      const currentScope = selectedScopeRef.current;
      if (currentScope !== 'ALL' && !pets.some((pet) => pet.id === currentScope)) {
        setSelectedScope('ALL');
        void loadStory(false, 'ALL');
      }
    } catch {
      if (requestId !== filterRequestRef.current) return;
      setFilterError(
        'Dog filters are unavailable. All-dogs Story still uses server-authorized history.'
      );
    } finally {
      if (requestId === filterRequestRef.current) setFilterLoading(false);
    }
  }, [loadStory, setSelectedScope]);

  useFocusEffect(
    useCallback(() => {
      void loadStory();
      void loadFilters();
      return () => {
        storyRequestRef.current += 1;
        filterRequestRef.current += 1;
      };
    }, [loadFilters, loadStory])
  );

  const chooseScope = useCallback(
    (scope: StoryScope) => {
      if (scope === selectedScopeRef.current) return;
      if (scope !== 'ALL' && !filterPets.some((pet) => pet.id === scope)) return;
      setSelectedScope(scope);
      setError(null);
      void loadStory(false, scope);
    },
    [filterPets, loadStory, setSelectedScope]
  );

  const activeDashboard = dashboard && dashboardScope === selectedScope ? dashboard : null;
  const moments = useMemo(
    () =>
      [...(activeDashboard?.moments ?? [])].sort(
        (a, b) => new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime()
      ),
    [activeDashboard]
  );

  if (loading && !dashboard) {
    return (
      <View style={styles.centered} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Gathering the moments that matter…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={() => void loadStory(true)} />
      }
    >
      <Text style={styles.eyebrow}>RELATIONSHIP MEMORY</Text>
      <Text style={styles.title}>Story</Text>
      <Text style={styles.subtitle}>
        A record of what you have actually lived together, not a feed you need to keep filling.
      </Text>

      <View style={styles.scopeSection}>
        <View style={styles.scopeHeadingRow}>
          <View>
            <Text style={styles.scopeEyebrow}>SHOWING</Text>
            <Text style={styles.scopeTitle}>
              {selectedScope === 'ALL'
                ? 'All your dogs'
                : (filterPets.find((pet) => pet.id === selectedScope)?.name ?? 'One dog')}
            </Text>
          </View>
          {filterLoading && <ActivityIndicator size="small" color={colors.primary[600]} />}
        </View>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.scopeRow}
        >
          <Pressable
            accessibilityRole="button"
            accessibilityState={{ selected: selectedScope === 'ALL' }}
            style={[styles.scopeChip, selectedScope === 'ALL' && styles.scopeChipSelected]}
            onPress={() => chooseScope('ALL')}
          >
            <Text
              style={[
                styles.scopeChipText,
                selectedScope === 'ALL' && styles.scopeChipTextSelected,
              ]}
            >
              All dogs
            </Text>
          </Pressable>
          {filterPets.map((pet) => {
            const selected = pet.id === selectedScope;
            return (
              <Pressable
                key={pet.id}
                accessibilityRole="button"
                accessibilityState={{ selected }}
                accessibilityLabel={`${selected ? 'Showing' : 'Show'} Story for ${pet.name}`}
                style={[styles.scopeChip, selected && styles.scopeChipSelected]}
                onPress={() => chooseScope(pet.id)}
              >
                <Text style={[styles.scopeChipText, selected && styles.scopeChipTextSelected]}>
                  {pet.name}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>
        <Text style={styles.scopeHint}>
          All dogs is one authorized household view. Choosing a dog narrows Story without changing
          Today or Compass.
        </Text>
        {filterError && <Text style={styles.filterError}>{filterError}</Text>}
      </View>

      {error && (
        <View style={styles.noticeCard} accessibilityRole="alert">
          <Ionicons name="cloud-offline-outline" size={20} color={colors.gray[600]} />
          <Text style={styles.noticeText}>{error}</Text>
        </View>
      )}

      {loading && !activeDashboard ? (
        <View style={styles.scopeLoadingCard} accessibilityRole="progressbar">
          <ActivityIndicator size="small" color={colors.primary[600]} />
          <Text style={styles.scopeLoadingText}>Opening this Story view…</Text>
        </View>
      ) : activeDashboard ? (
        <>
          <View style={styles.statsCard}>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{activeDashboard.stats.activities}</Text>
              <Text style={styles.statLabel}>Activities</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{activeDashboard.stats.memories}</Text>
              <Text style={styles.statLabel}>Memories</Text>
            </View>
            <View style={styles.stat}>
              <Text style={styles.statValue}>{activeDashboard.stats.namedPlaces}</Text>
              <Text style={styles.statLabel}>Places</Text>
            </View>
          </View>

          {activeDashboard.milestones.length > 0 && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Milestones</Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.milestonesRow}
              >
                {activeDashboard.milestones.slice(0, 8).map((milestone) => (
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
            <Text style={styles.sectionSubtitle}>
              Recent moments from activity, care, and memories.
            </Text>

            {moments.length === 0 ? (
              <View style={styles.emptyCard}>
                <Ionicons name="paw-outline" size={28} color={colors.primary[600]} />
                <Text style={styles.emptyTitle}>This Story view is just beginning.</Text>
                <Text style={styles.emptyText}>
                  Nothing is missing or overdue. Woof will keep useful moments without turning
                  everyday life into homework.
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
            Story coverage: {activeDashboard.stats.coverage.toLowerCase()}. Woof may intentionally
            show a bounded recent history rather than pretending this is every moment you have
            shared.
          </Text>
        </>
      ) : null}
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
  scopeSection: {
    marginTop: 18,
    paddingVertical: 14,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: colors.gray[200],
  },
  scopeHeadingRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  scopeEyebrow: {
    color: colors.text.secondary,
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 1.2,
  },
  scopeTitle: { marginTop: 2, color: colors.text.primary, fontSize: 16, fontWeight: '800' },
  scopeRow: { gap: 8, paddingTop: 10, paddingRight: 18 },
  scopeChip: {
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: 14,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[300],
    backgroundColor: '#ffffff',
  },
  scopeChipSelected: { borderColor: colors.primary[300], backgroundColor: colors.primary[100] },
  scopeChipText: { color: colors.gray[700], fontSize: 13, fontWeight: '700' },
  scopeChipTextSelected: { color: colors.primary[900] },
  scopeHint: { marginTop: 8, color: colors.text.secondary, fontSize: 11, lineHeight: 17 },
  filterError: { marginTop: 6, color: colors.text.secondary, fontSize: 10, lineHeight: 15 },
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
  scopeLoadingCard: {
    minHeight: 84,
    marginTop: 18,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  scopeLoadingText: { color: colors.text.secondary, fontSize: 12 },
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
  milestoneDescription: {
    marginTop: 4,
    color: colors.text.secondary,
    fontSize: 12,
    lineHeight: 17,
  },
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
  emptyText: {
    marginTop: 6,
    color: colors.text.secondary,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
  },
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
