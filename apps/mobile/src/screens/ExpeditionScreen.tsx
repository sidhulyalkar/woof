import React, { useCallback, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import type { StackScreenProps } from '@react-navigation/stack';
import { expeditionApi, type ExpeditionProjection } from '../api/expeditions';
import { socialAdventureApi, type PacksCatalog } from '../api/social-adventure';
import { ExpeditionWorldView } from '../components/community/ExpeditionWorldView';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Expedition'>;

type SelectedScope = 'GLOBAL' | string;

export default function ExpeditionScreen({ navigation }: Props) {
  const [globalProjection, setGlobalProjection] = useState<ExpeditionProjection | null>(null);
  const [packProjection, setPackProjection] = useState<ExpeditionProjection | null>(null);
  const [catalog, setCatalog] = useState<PacksCatalog | null>(null);
  const [selectedScope, setSelectedScope] = useState<SelectedScope>('GLOBAL');
  const [loading, setLoading] = useState(true);
  const [packLoading, setPackLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const packRequestRef = useRef(0);
  const selectedScopeRef = useRef<SelectedScope>('GLOBAL');

  const joinedPacks = useMemo(() => catalog?.packs.filter((pack) => pack.joined) ?? [], [catalog]);

  const applySelectedScope = useCallback((scope: SelectedScope) => {
    selectedScopeRef.current = scope;
    setSelectedScope(scope);
  }, []);

  const loadPackProjection = useCallback(
    async (packId: string, packs: PacksCatalog | null) => {
      const joinedPack = packs?.packs.find((pack) => pack.id === packId && pack.joined);
      if (!joinedPack) {
        packRequestRef.current += 1;
        setPackProjection(null);
        setPackLoading(false);
        applySelectedScope('GLOBAL');
        return;
      }

      const requestId = ++packRequestRef.current;
      setPackLoading(true);
      try {
        const response = await expeditionApi.pack(joinedPack.id);
        if (requestId !== packRequestRef.current) return;
        if (response.scope !== 'PACK' || response.pack?.id !== joinedPack.id) {
          setPackProjection(null);
          setError('That Pack Expedition did not match the server-confirmed Pack, so Woof hid it.');
          return;
        }
        setPackProjection(response);
      } catch {
        if (requestId !== packRequestRef.current) return;
        setPackProjection(null);
        setError(
          'That Pack Expedition is unavailable. Woof will not estimate shared progress from local activity or league data.'
        );
      } finally {
        if (requestId === packRequestRef.current) setPackLoading(false);
      }
    },
    [applySelectedScope]
  );

  const load = useCallback(
    async (refresh = false) => {
      if (refresh) setRefreshing(true);
      else setLoading(true);

      const [globalResult, packsResult] = await Promise.allSettled([
        expeditionApi.global(),
        socialAdventureApi.packs(),
      ]);

      const unavailable: string[] = [];

      if (globalResult.status === 'fulfilled') {
        if (globalResult.value.scope === 'GLOBAL') setGlobalProjection(globalResult.value);
        else unavailable.push('global Expedition');
      } else {
        unavailable.push('global Expedition');
      }

      if (packsResult.status === 'fulfilled') {
        setCatalog(packsResult.value);
        const currentScope = selectedScopeRef.current;
        if (currentScope !== 'GLOBAL') {
          const stillJoined = packsResult.value.packs.some(
            (pack) => pack.id === currentScope && pack.joined
          );
          if (stillJoined) await loadPackProjection(currentScope, packsResult.value);
          else {
            packRequestRef.current += 1;
            setPackProjection(null);
            setPackLoading(false);
            applySelectedScope('GLOBAL');
          }
        }
      } else {
        unavailable.push('Pack membership');
      }

      if (unavailable.length === 0) setError(null);
      else
        setError(
          `${unavailable.join(' and ')} could not refresh. Previously loaded server projections remain visible where available; Woof did not infer missing progress or membership.`
        );

      setLoading(false);
      setRefreshing(false);
    },
    [applySelectedScope, loadPackProjection]
  );

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const selectGlobal = useCallback(() => {
    packRequestRef.current += 1;
    setPackProjection(null);
    setPackLoading(false);
    setError(null);
    applySelectedScope('GLOBAL');
  }, [applySelectedScope]);

  const selectPack = useCallback(
    (packId: string) => {
      const joinedPack = catalog?.packs.find((pack) => pack.id === packId && pack.joined);
      if (!joinedPack) {
        setError('Woof will only open Expedition views for Packs the server says you joined.');
        return;
      }
      setError(null);
      setPackProjection(null);
      applySelectedScope(joinedPack.id);
      void loadPackProjection(joinedPack.id, catalog);
    },
    [applySelectedScope, catalog, loadPackProjection]
  );

  if (loading && !globalProjection) {
    return (
      <View style={styles.loading} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening this week’s shared world…</Text>
      </View>
    );
  }

  return (
    <ExpeditionWorldView
      globalProjection={globalProjection}
      packProjection={packProjection}
      joinedPacks={joinedPacks}
      selectedScope={selectedScope}
      packLoading={packLoading}
      refreshing={refreshing}
      error={error}
      onRefresh={() => void load(true)}
      onSelectGlobal={selectGlobal}
      onSelectPack={selectPack}
      onOpenSkillcraft={() => navigation.navigate('Skillcraft')}
      onOpenPacks={() => navigation.navigate('Packs')}
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
