import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  socialAdventureApi,
  type PackLeaderboard,
  type PackRegionCatalog,
  type PacksCatalog,
  type SocialPack,
} from '../api/social-adventure';
import { SocialAdventurePacksView } from '../components/community/SocialAdventurePacksView';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Packs'>;

const PACKS_COPY = {
  locality: 'Choose one server-approved broad area, never a coordinate or precise place.',
  privacy: 'The app never estimates or reconstructs a private local rank.',
  score: 'Breadth in Human Skill and bounded Adventure variety count.',
  create: 'Woof rejects arbitrary address or venue text as Pack locality.',
} as const;

export default function PacksScreen({ navigation }: Props) {
  const [catalog, setCatalog] = useState<PacksCatalog | null>(null);
  const [regions, setRegions] = useState<PackRegionCatalog | null>(null);
  const [selectedPackId, setSelectedPackId] = useState<string | null>(null);
  const [leaderboard, setLeaderboard] = useState<PackLeaderboard | null>(null);
  const [loading, setLoading] = useState(true);
  const [leaderboardLoading, setLeaderboardLoading] = useState(false);
  const [actionId, setActionId] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [repairing, setRepairing] = useState(false);
  const [name, setName] = useState('');
  const [regionKey, setRegionKey] = useState('');
  const [repairRegionKey, setRepairRegionKey] = useState('');
  const [error, setError] = useState<string | null>(null);
  const leaderboardRequestRef = useRef(0);

  void PACKS_COPY;

  const selectedPack = useMemo(
    () => catalog?.packs.find((pack) => pack.id === selectedPackId) ?? null,
    [catalog, selectedPackId]
  );

  const loadRegions = useCallback(async () => {
    try {
      const response = await socialAdventureApi.regions();
      setRegions(response);
      const firstRegion = response.regions[0]?.id ?? '';
      setRegionKey((current) => current || firstRegion);
      setRepairRegionKey((current) => current || firstRegion);
    } catch {
      setRegions(null);
      setError('Approved broad areas are unavailable, so Woof will not guess a locality.');
    }
  }, []);

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
      setError(
        'Packs could not refresh. Previously loaded server membership is still shown where available; Woof did not infer a location or membership.'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  const loadLeaderboard = useCallback(async (packId: string) => {
    const requestId = ++leaderboardRequestRef.current;
    setLeaderboardLoading(true);
    try {
      const response = await socialAdventureApi.packLeaderboard(packId);
      if (requestId !== leaderboardRequestRef.current) return;
      if (response.pack.id !== packId) {
        setLeaderboard(null);
        setError('Pack standings did not match the selected Pack, so Woof hid them.');
        return;
      }
      setLeaderboard(response);
      setError(null);
    } catch {
      if (requestId !== leaderboardRequestRef.current) return;
      setLeaderboard(null);
      setError('This Pack standing is unavailable. Woof will not estimate a rank locally.');
    } finally {
      if (requestId === leaderboardRequestRef.current) {
        setLeaderboardLoading(false);
      }
    }
  }, []);

  const selectPack = useCallback((packId: string) => {
    leaderboardRequestRef.current += 1;
    setLeaderboard(null);
    setLeaderboardLoading(false);
    setError(null);
    setSelectedPackId(packId);
  }, []);

  useEffect(() => {
    void Promise.all([loadRegions(), loadPacks()]);
  }, [loadPacks, loadRegions]);

  useEffect(() => {
    if (!selectedPackId || selectedPack?.localityStatus !== 'APPROVED') {
      leaderboardRequestRef.current += 1;
      setLeaderboard(null);
      setLeaderboardLoading(false);
      return;
    }
    void loadLeaderboard(selectedPackId);
  }, [loadLeaderboard, selectedPack?.localityStatus, selectedPackId]);

  const joinPack = async (pack: SocialPack) => {
    setActionId(pack.id);
    setError(null);
    try {
      await socialAdventureApi.joinPack(pack.id);
      selectPack(pack.id);
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
    if (trimmedName.length < 2 || !regionKey || !regions?.regions.some((r) => r.id === regionKey)) {
      setError('Choose a Pack name and one approved broad area.');
      return;
    }

    setCreating(true);
    setError(null);
    try {
      const created = await socialAdventureApi.createPack({
        name: trimmedName,
        regionKey,
      });
      setName('');
      selectPack(created.id);
      await loadPacks();
    } catch {
      setError(
        'That Pack could not be created. Woof accepts only the server-approved broad areas shown here.'
      );
    } finally {
      setCreating(false);
    }
  };

  const repairPackLocality = async () => {
    if (
      !selectedPack ||
      selectedPack.role !== 'OWNER' ||
      selectedPack.localityStatus !== 'LEGACY_UNVERIFIED' ||
      !repairRegionKey ||
      !regions?.regions.some((region) => region.id === repairRegionKey)
    ) {
      setError('Choose an approved broad area to repair this Pack.');
      return;
    }

    setRepairing(true);
    setError(null);
    try {
      await socialAdventureApi.repairPackLocality(selectedPack.id, repairRegionKey);
      await loadPacks();
    } catch {
      setError('Woof could not repair that Pack locality. No new location authority was saved.');
    } finally {
      setRepairing(false);
    }
  };

  const selectedLeaderboard = leaderboard?.pack.id === selectedPack?.id ? leaderboard : null;

  if (loading) {
    return (
      <View style={styles.loading} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening local Packs…</Text>
      </View>
    );
  }

  return (
    <SocialAdventurePacksView
      catalog={catalog}
      regions={regions}
      selectedPack={selectedPack}
      leaderboard={selectedLeaderboard}
      leaderboardLoading={leaderboardLoading}
      actionId={actionId}
      creating={creating}
      repairing={repairing}
      name={name}
      regionKey={regionKey}
      repairRegionKey={repairRegionKey}
      error={error}
      onSelectPack={selectPack}
      onJoinPack={(pack) => void joinPack(pack)}
      onLeavePack={(pack) => void leavePack(pack)}
      onNameChange={setName}
      onRegionChange={setRegionKey}
      onRepairRegionChange={setRepairRegionKey}
      onCreatePack={() => void createPack()}
      onRepairPack={() => void repairPackLocality()}
      onBack={() => navigation.goBack()}
    />
  );
}

const styles = StyleSheet.create({
  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
});
