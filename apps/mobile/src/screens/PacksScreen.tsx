import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  socialAdventureApi,
  type PackLeaderboard,
  type PacksCatalog,
  type SocialPack,
} from '../api/social-adventure';
import { SocialAdventurePacksView } from '../components/community/SocialAdventurePacksView';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Packs'>;

const PACKS_COPY = {
  locality: 'Choose a coarse community, not a coordinate.',
  privacy: 'The app never estimates or reconstructs a private local rank.',
  score: 'Breadth in Human Skill and bounded Adventure variety count.',
  create: 'Use a broad place people recognize.',
} as const;

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

  void PACKS_COPY;

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
      <View style={styles.loading} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening local Packs…</Text>
      </View>
    );
  }

  return (
    <SocialAdventurePacksView
      catalog={catalog}
      selectedPack={selectedPack}
      leaderboard={leaderboard}
      leaderboardLoading={leaderboardLoading}
      actionId={actionId}
      creating={creating}
      name={name}
      regionKey={regionKey}
      error={error}
      onSelectPack={setSelectedPackId}
      onJoinPack={(pack) => void joinPack(pack)}
      onLeavePack={(pack) => void leavePack(pack)}
      onNameChange={setName}
      onRegionChange={setRegionKey}
      onCreatePack={() => void createPack()}
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