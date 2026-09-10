import React, { useCallback, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  expeditionApi,
  type ExpeditionJournal,
  type ExpeditionProjection,
} from '../api/expeditions';
import { socialAdventureApi, type PacksCatalog } from '../api/social-adventure';
import { ExpeditionWorldView } from '../components/community/ExpeditionWorldView';
import type { RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = StackScreenProps<RootStackParamList, 'Expedition'>;

type SelectedScope = 'GLOBAL' | string;

type PackLoadResult =
  | { status: 'ok' }
  | { status: 'stale' }
  | { status: 'error'; message: string };

export default function ExpeditionScreen({ navigation }: Props) {
  const [globalProjection, setGlobalProjection] = useState<ExpeditionProjection | null>(null);
  const [packProjection, setPackProjection] = useState<ExpeditionProjection | null>(null);
  const [journal, setJournal] = useState<ExpeditionJournal | null>(null);
  const [catalog, setCatalog] = useState<PacksCatalog | null>(null);
  const [selectedScope, setSelectedScope] = useState<SelectedScope>('GLOBAL');
  const [loading, setLoading] = useState(true);
  const [packLoading, setPackLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [worldError, setWorldError] = useState<string | null>(null);
  const [journalError, setJournalError] = useState<string | null>(null);
  const loadRequestRef = useRef(0);
  const packRequestRef = useRef(0);
  const selectedScopeRef = useRef<SelectedScope>('GLOBAL');
  const journalRef = useRef<ExpeditionJournal | null>(null);

  const joinedPacks = useMemo(() => catalog?.packs.filter((pack) => pack.joined) ?? [], [catalog]);

  const applySelectedScope = useCallback((scope: SelectedScope) => {
    selectedScopeRef.current = scope;
    setSelectedScope(scope);
  }, []);

  const loadPackProjection = useCallback(
    async (packId: string, packs: PacksCatalog | null): Promise<PackLoadResult> => {
      const joinedPack = packs?.packs.find((pack) => pack.id === packId && pack.joined);
      if (!joinedPack) {
        packRequestRef.current += 1;
        setPackProjection(null);
        setPackLoading(false);
        applySelectedScope('GLOBAL');
        return {
          status: 'error',
          message: 'Woof will only open Expedition views for Packs you have joined.',
        };
      }

      const requestId = ++packRequestRef.current;
      setPackLoading(true);
      try {
        const response = await expeditionApi.pack(joinedPack.id);
        if (requestId !== packRequestRef.current || selectedScopeRef.current !== joinedPack.id) {
          return { status: 'stale' };
        }
        if (response.scope !== 'PACK' || response.pack?.id !== joinedPack.id) {
          setPackProjection(null);
          return {
            status: 'error',
            message: 'That Pack Expedition did not match the Pack you opened, so Woof hid it.',
          };
        }
        setPackProjection(response);
        return { status: 'ok' };
      } catch {
        if (requestId !== packRequestRef.current || selectedScopeRef.current !== joinedPack.id) {
          return { status: 'stale' };
        }
        setPackProjection(null);
        return {
          status: 'error',
          message:
            'That Pack Expedition is unavailable. Woof will leave the shared world blank rather than guess.',
        };
      } finally {
        if (requestId === packRequestRef.current) setPackLoading(false);
      }
    },
    [applySelectedScope]
  );

  const load = useCallback(
    async (refresh = false) => {
      const loadRequestId = ++loadRequestRef.current;
      if (refresh) setRefreshing(true);
      else setLoading(true);

      const [globalResult, packsResult, journalResult] = await Promise.allSettled([
        expeditionApi.global(),
        socialAdventureApi.packs(),
        expeditionApi.journal(),
      ]);

      if (loadRequestId !== loadRequestRef.current) return;

      const unavailableWorld: string[] = [];
      let packLoadResult: PackLoadResult | null = null;

      if (globalResult.status === 'fulfilled') {
        if (globalResult.value.scope === 'GLOBAL') setGlobalProjection(globalResult.value);
        else unavailableWorld.push('shared Expedition');
      } else {
        unavailableWorld.push('shared Expedition');
      }

      if (packsResult.status === 'fulfilled') {
        setCatalog(packsResult.value);
        const currentScope = selectedScopeRef.current;
        if (currentScope !== 'GLOBAL') {
          const stillJoined = packsResult.value.packs.some(
            (pack) => pack.id === currentScope && pack.joined
          );
          if (stillJoined) packLoadResult = await loadPackProjection(currentScope, packsResult.value);
          else {
            packRequestRef.current += 1;
            setPackProjection(null);
            setPackLoading(false);
            applySelectedScope('GLOBAL');
          }
        }
      } else {
        unavailableWorld.push('Pack list');
      }

      if (loadRequestId !== loadRequestRef.current) return;

      if (journalResult.status === 'fulfilled' && journalResult.value.scope === 'GLOBAL') {
        journalRef.current = journalResult.value;
        setJournal(journalResult.value);
        setJournalError(null);
      } else {
        setJournalError(
          journalRef.current
            ? 'Could not refresh field notes. Showing your last verified pages.'
            : 'Field notes could not refresh. Your shared world is still available; Woof will leave history blank rather than guess.'
        );
      }

      if (packLoadResult?.status === 'error') {
        setWorldError(packLoadResult.message);
      } else if (unavailableWorld.length === 0) {
        setWorldError(null);
      } else {
        setWorldError(
          `${unavailableWorld.join(' and ')} could not refresh. Previously loaded shared-world data stays visible where available; Woof will not invent the missing view.`
        );
      }

      setLoading(false);
      setRefreshing(false);
    },
    [applySelectedScope, loadPackProjection]
  );

  useFocusEffect(
    useCallback(() => {
      void load();
      return () => {
        loadRequestRef.current += 1;
        packRequestRef.current += 1;
      };
    }, [load])
  );

  const selectGlobal = useCallback(() => {
    packRequestRef.current += 1;
    setPackProjection(null);
    setPackLoading(false);
    setWorldError(null);
    applySelectedScope('GLOBAL');
  }, [applySelectedScope]);

  const selectPack = useCallback(
    (packId: string) => {
      const joinedPack = catalog?.packs.find((pack) => pack.id === packId && pack.joined);
      if (!joinedPack) {
        setWorldError('Woof will only open Expedition views for Packs you have joined.');
        return;
      }
      setWorldError(null);
      setPackProjection(null);
      applySelectedScope(joinedPack.id);
      void loadPackProjection(joinedPack.id, catalog).then((result) => {
        if (result.status === 'stale' || selectedScopeRef.current !== joinedPack.id) return;
        setWorldError(result.status === 'error' ? result.message : null);
      });
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
      journal={journal}
      journalError={journalError}
      joinedPacks={joinedPacks}
      selectedScope={selectedScope}
      packLoading={packLoading}
      refreshing={refreshing}
      error={worldError}
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
