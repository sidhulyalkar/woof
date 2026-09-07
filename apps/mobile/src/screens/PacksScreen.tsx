import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  socialAdventureApi,
  type PackLeaderboard,
  type PacksCatalog,
  type SocialPack,
} from '../api/social-adventure';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Packs'>;

const normalizeRegionKey = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');

export default function PacksScreen({ navigation }: Props) {
  const [catalog, setCatalog] = useState<PacksCatalog | null>(null);
  const [selectedPackId, setSelectedPackId] = useState<string | null>(null);
  const [leaderboard, setLeaderboard] = useState<PackLeaderboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [actionId, setActionId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [regionKey, setRegionKey] = useState('');
  const [error, setError] = useState<string | null>(null);

  const selectedPack = useMemo(
    () => catalog?.packs.find((pack) => pack.id === selectedPackId) ?? null,
    [catalog, selectedPackId]
  );

  const loadPacks = useCallback(async () => {
    try {
      const response = await socialAdventureApi.packs();
      setCatalog(response);
      setSelectedPackId((current) => {
        if (current && response.packs.some((pack) => pack.id === current)) return current;
        return response.packs.find((pack) => pack.joined)?.id ?? response.packs[0]?.id ?? null;
      });
      setError(null);
    } catch {
      setCatalog(null);
      setSelectedPackId(null);
      setLeaderboard(null);
      setError('Packs are unavailable right now. Woof did not infer a location or membership.');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadLeaderboard = useCallback(async (packId: string) => {
    setLeaderboardLoading(true);
    try {
      const response = await socialAdventureApi.packLeaderboard(packId);
      setLeaderboard(response);
      setError(null);
    } catch {
      setLeaderboard(null);
      setError('This Pack standing is unavailable. Woof will not estimate a rank locally.');
    } finally {
      setLeaderboardLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadPacks();
  }, [loadPacks]);

  useEffect(() => {
    if (!selectedPackId) {
      setLeaderboard(null);
      return;
    }
    void loadLeaderboard(selectedPackId);
  }, [loadLeaderboard, selectedPackId]);

  const joinPack = async (pack: SocialPack) => {
    setActionId(pack.id);
    setError(null);
    try {
      await socialAdventureApi.joinPack(pack.id);
      setSelectedPackId(pack.id);
      await loadPacks();
    } catch {
      setError('Woof could not join that Pack. No membership was changed.');
    } finally {
      setActionId(null);
    }
  };

  const leavePack = async (pack: SocialPack) => {
    if (pack.role === 'OWNER') {
      setError('Pack owners must transfer or retire their Pack before leaving.');
      return;
    }

    setActionId(pack.id);
    setError(null);
    try {
      await socialAdventureApi.leavePack(pack.id);
      await loadPacks();
    } catch {
      setError('Woof could not leave that Pack. Your existing membership stays unchanged.');
    } finally {
      setActionId(null);
    }
  };

  const createPack = async () => {
    const trimmedName = name.trim();
    const normalizedRegion = normalizeRegionKey(regionKey);
    if (trimmedName.length < 2 || normalizedRegion.length < 2) {
      setError('Use a Pack name and a broad region label such as south-bay-ca.');
      return;
    }

    setCreating(true);
    setError(null);
    try {
      const created = await socialAdventureApi.createPack({
        name: trimmedName,
        regionKey: normalizedRegion,
      });
      setName('');
      setRegionKey('');
      setSelectedPackId(created.id);
      await loadPacks();
    } catch {
      setError(
        'That Pack could not be created. Use a coarse region, not an address, route, or exact meetup point.'
      );
    } finally {
      setCreating(false);
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening local Packs…</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.screen}
      contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="map-outline" size={23} color={colors.primary[700]} />
        </View>
        <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
        <Text style={styles.title}>Local Packs without tracking you.</Text>
        <Text style={styles.intro}>
          Choose a coarse community, not a coordinate. Pack rank never uses your home, route
          endpoints, live GPS, health, mileage, or a dog's performance.
        </Text>
        <View style={styles.boundaryCard}>
          <Ionicons name="shield-checkmark-outline" size={20} color={colors.success.dark} />
          <Text style={styles.boundaryText}>
            Local standings stay hidden until the server says the cohort is large enough. The app
            never estimates or reconstructs a private local rank.
          </Text>
        </View>
      </View>

      <View style={styles.sectionHeader}>
        <View>
          <Text style={styles.sectionEyebrow}>OPT-IN COMMUNITIES</Text>
          <Text style={styles.sectionTitle}>Find a Pack</Text>
        </View>
        <Ionicons name="people-outline" size={22} color={colors.primary[600]} />
      </View>

      {!catalog?.packs.length ? (
        <View style={styles.emptyCard}>
          <Text style={styles.emptyTitle}>No local Packs yet.</Text>
          <Text style={styles.emptyText}>A quiet map is valid. You can start a broad-area Pack below.</Text>
        </View>
      ) : (
        <View style={styles.packList}>
          {catalog.packs.map((pack) => {
            const selected = pack.id === selectedPackId;
            const busy = actionId === pack.id;
            return (
              <Pressable
                key={pack.id}
                accessibilityRole="button"
                accessibilityLabel={`Open ${pack.name} Pack`}
                style={[styles.packCard, selected && styles.packCardSelected]}
                onPress={() => setSelectedPackId(pack.id)}
              >
                <View style={styles.packCopy}>
                  <Text style={styles.packName}>{pack.name}</Text>
                  <Text style={styles.packMeta}>
                    {pack.regionKey ?? 'coarse region'} · {pack.memberCount}{' '}
                    {pack.memberCount === 1 ? 'member' : 'members'}
                  </Text>
                </View>
                {pack.role === 'OWNER' ? (
                  <View style={styles.joinedBadge}>
                    <Text style={styles.joinedText}>Owner</Text>
                  </View>
                ) : pack.joined ? (
                  <Pressable
                    accessibilityRole="button"
                    accessibilityLabel={`Leave ${pack.name}`}
                    disabled={busy}
                    style={styles.secondaryAction}
                    onPress={() => void leavePack(pack)}
                  >
                    {busy ? (
                      <ActivityIndicator size="small" color={colors.primary[700]} />
                    ) : (
                      <Text style={styles.secondaryActionText}>Leave</Text>
                    )}
                  </Pressable>
                ) : (
                  <Pressable
                    accessibilityRole="button"
                    accessibilityLabel={`Join ${pack.name}`}
                    disabled={busy}
                    style={styles.secondaryAction}
                    onPress={() => void joinPack(pack)}
                  >
                    {busy ? (
                      <ActivityIndicator size="small" color={colors.primary[700]} />
                    ) : (
                      <Text style={styles.secondaryActionText}>Join</Text>
                    )}
                  </Pressable>
                )}
              </Pressable>
            );
          })}
        </View>
      )}

      {selectedPack && (
        <View style={styles.leagueSection}>
          <Text style={styles.sectionEyebrow}>PACK LEAGUE</Text>
          <Text style={styles.sectionTitle}>Human-side standings</Text>
          <Text style={styles.leagueIntro}>
            Breadth in Human Skill and bounded Adventure variety count. Repetition volume, likes,
            missed days, health, and exercise intensity do not.
          </Text>

          {leaderboardLoading ? (
            <View style={styles.inlineLoading} accessibilityRole="progressbar">
              <ActivityIndicator size="small" color={colors.primary[600]} />
            </View>
          ) : leaderboard && !leaderboard.cohortReady ? (
            <View style={styles.cohortCard}>
              <Ionicons name="shield-outline" size={22} color={colors.primary[700]} />
              <View style={styles.cohortCopy}>
                <Text style={styles.cohortTitle}>Building a privacy-safe cohort</Text>
                <Text style={styles.cohortText}>
                  {leaderboard.message ?? 'Standings remain hidden until the server cohort is ready.'}
                </Text>
                <Text style={styles.cohortCount}>
                  {leaderboard.pack.memberCount}/{leaderboard.minimumCohort} members
                </Text>
              </View>
            </View>
          ) : leaderboard?.cohortReady ? (
            <View style={styles.standings}>
              {leaderboard.entries.map((entry) => (
                <View key={entry.userId} style={styles.standingRow}>
                  <View style={styles.rankBadge}>
                    <Text style={styles.rankText}>{entry.rank}</Text>
                  </View>
                  <View style={styles.standingCopy}>
                    <Text style={styles.standingHandle}>@{entry.handle}</Text>
                    <Text style={styles.standingMeta}>
                      {entry.components.humanSkill.score} skill ·{' '}
                      {entry.components.adventureVariety.pathways.length} pathways
                    </Text>
                  </View>
                  <Text style={styles.standingScore}>{entry.score}</Text>
                </View>
              ))}
            </View>
          ) : null}
        </View>
      )}

      <View style={styles.createCard}>
        <View style={styles.createHeading}>
          <Ionicons name="add-circle-outline" size={22} color={colors.primary[700]} />
          <View style={styles.createHeadingCopy}>
            <Text style={styles.createTitle}>Start a coarse-locality Pack</Text>
            <Text style={styles.createText}>
              Use a broad place people recognize. Do not enter an address, apartment complex,
              school, route, or exact meetup point.
            </Text>
          </View>
        </View>

        <Text style={styles.fieldLabel}>Pack name</Text>
        <TextInput
          value={name}
          onChangeText={setName}
          maxLength={64}
          placeholder="South Bay Adventure Pack"
          placeholderTextColor={colors.gray[400]}
          style={styles.input}
        />

        <Text style={styles.fieldLabel}>Coarse region</Text>
        <TextInput
          value={regionKey}
          onChangeText={setRegionKey}
          autoCapitalize="none"
          autoCorrect={false}
          maxLength={64}
          placeholder="south-bay-ca"
          placeholderTextColor={colors.gray[400]}
          style={styles.input}
        />

        <Pressable
          accessibilityRole="button"
          disabled={creating}
          style={[styles.primaryButton, creating && styles.disabled]}
          onPress={() => void createPack()}
        >
          {creating ? (
            <ActivityIndicator color="#ffffff" />
          ) : (
            <Text style={styles.primaryButtonText}>Create Pack</Text>
          )}
        </Pressable>
      </View>

      {catalog && (
        <Text style={styles.locationContract}>
          Server location contract: {catalog.locationContract}. Minimum cohort:{' '}
          {catalog.localMinimumCohort}.
        </Text>
      )}

      {error && (
        <View style={styles.errorCard} accessibilityRole="alert">
          <Ionicons name="alert-circle-outline" size={18} color={colors.error.dark} />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      <Pressable
        accessibilityRole="button"
        style={styles.backButton}
        onPress={() => navigation.goBack()}
      >
        <Text style={styles.backButtonText}>Back to Community</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  content: { padding: 18, paddingBottom: 50 },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  hero: {
    borderRadius: 24,
    padding: 20,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  heroIcon: {
    width: 46,
    height: 46,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
    marginBottom: 18,
  },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.3 },
  title: { marginTop: 5, color: colors.gray[900], fontSize: 27, lineHeight: 34, fontWeight: '800' },
  intro: { marginTop: 9, color: colors.gray[700], fontSize: 14, lineHeight: 21 },
  boundaryCard: {
    marginTop: 16,
    borderRadius: 15,
    padding: 13,
    flexDirection: 'row',
    gap: 9,
    backgroundColor: colors.success.light,
  },
  boundaryText: { flex: 1, color: colors.success.dark, fontSize: 12, lineHeight: 18 },
  sectionHeader: {
    marginTop: 28,
    marginBottom: 11,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  sectionEyebrow: {
    color: colors.primary[700],
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  emptyCard: { padding: 18, borderRadius: 17, backgroundColor: '#ffffff' },
  emptyTitle: { color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  emptyText: { marginTop: 5, color: colors.gray[600], fontSize: 13, lineHeight: 19 },
  packList: { gap: 9 },
  packCard: {
    minHeight: 76,
    padding: 14,
    borderRadius: 17,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: '#ffffff',
    flexDirection: 'row',
    alignItems: 'center',
  },
  packCardSelected: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  packCopy: { flex: 1, paddingRight: 10 },
  packName: { color: colors.gray[900], fontSize: 15, fontWeight: '800' },
  packMeta: { marginTop: 4, color: colors.gray[600], fontSize: 12 },
  joinedBadge: {
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: colors.primary[100],
  },
  joinedText: { color: colors.primary[800], fontSize: 11, fontWeight: '800' },
  secondaryAction: {
    minWidth: 64,
    minHeight: 38,
    paddingHorizontal: 12,
    borderRadius: 11,
    borderWidth: 1,
    borderColor: colors.primary[200],
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#ffffff',
  },
  secondaryActionText: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  leagueSection: { marginTop: 30 },
  leagueIntro: { marginTop: 7, color: colors.gray[600], fontSize: 13, lineHeight: 19 },
  inlineLoading: { minHeight: 76, alignItems: 'center', justifyContent: 'center' },
  cohortCard: {
    marginTop: 12,
    padding: 16,
    borderRadius: 17,
    backgroundColor: colors.primary[50],
    flexDirection: 'row',
    gap: 11,
  },
  cohortCopy: { flex: 1 },
  cohortTitle: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  cohortText: { marginTop: 4, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  cohortCount: { marginTop: 8, color: colors.primary[700], fontSize: 12, fontWeight: '800' },
  standings: { marginTop: 12, gap: 8 },
  standingRow: {
    minHeight: 64,
    padding: 11,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: '#ffffff',
    flexDirection: 'row',
    alignItems: 'center',
  },
  rankBadge: {
    width: 36,
    height: 36,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[100],
  },
  rankText: { color: colors.primary[800], fontSize: 13, fontWeight: '900' },
  standingCopy: { flex: 1, marginLeft: 11 },
  standingHandle: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  standingMeta: { marginTop: 3, color: colors.gray[600], fontSize: 11 },
  standingScore: { color: colors.primary[700], fontSize: 18, fontWeight: '900' },
  createCard: {
    marginTop: 30,
    padding: 18,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: '#ffffff',
  },
  createHeading: { flexDirection: 'row', gap: 10 },
  createHeadingCopy: { flex: 1 },
  createTitle: { color: colors.gray[900], fontSize: 16, fontWeight: '800' },
  createText: { marginTop: 4, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  fieldLabel: { marginTop: 15, marginBottom: 6, color: colors.gray[800], fontSize: 12, fontWeight: '800' },
  input: {
    minHeight: 47,
    borderWidth: 1,
    borderColor: colors.gray[300],
    borderRadius: 12,
    paddingHorizontal: 12,
    color: colors.gray[900],
    backgroundColor: colors.gray[50],
  },
  primaryButton: {
    minHeight: 49,
    borderRadius: 13,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
    marginTop: 17,
  },
  primaryButtonText: { color: '#ffffff', fontSize: 14, fontWeight: '800' },
  disabled: { opacity: 0.55 },
  locationContract: {
    marginTop: 13,
    color: colors.gray[500],
    fontSize: 11,
    lineHeight: 16,
    textAlign: 'center',
  },
  errorCard: {
    marginTop: 14,
    padding: 12,
    borderRadius: 12,
    flexDirection: 'row',
    gap: 8,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  backButton: { minHeight: 45, marginTop: 20, alignItems: 'center', justifyContent: 'center' },
  backButtonText: { color: colors.primary[700], fontSize: 13, fontWeight: '800' },
});
