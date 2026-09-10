import React, { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { CompositeScreenProps } from '@react-navigation/native';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';
import type { StackScreenProps } from '@react-navigation/stack';
import {
  socialAdventureApi,
  type GlobalLeaderboard,
  type SocialAdventureMe,
  type SocialAdventurePost,
  type SocialAdventureReaction,
} from '../api/social-adventure';
import { SocialAdventureCommunityView } from '../components/community/SocialAdventureCommunityView';
import type { MainTabParamList, RootStackParamList } from '../navigation/AppNavigator';
import { colors } from '../theme/tokens';

type Props = CompositeScreenProps<
  BottomTabScreenProps<MainTabParamList, 'Community'>,
  StackScreenProps<RootStackParamList>
>;

const COMMUNITY_COPY = {
  hero: 'You compete. Your dog does not.',
  privateScore: 'Your score is private by default.',
  reactions: 'Reactions build culture, not rank.',
  sharing: 'Nothing posts automatically.',
  emptyLeague: 'An empty podium is allowed.',
  privateAction: 'Make my rank private',
  publicAction: 'Join global league',
} as const;

export default function FeedScreen({ navigation }: Props) {
  const [posts, setPosts] = useState<SocialAdventurePost[]>([]);
  const [me, setMe] = useState<SocialAdventureMe | null>(null);
  const [leaderboard, setLeaderboard] = useState<GlobalLeaderboard | null>(null);
  const [feedPrivacy, setFeedPrivacy] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [preferenceSaving, setPreferenceSaving] = useState(false);
  const [reactionSaving, setReactionSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  void COMMUNITY_COPY;

  const loadCommunity = useCallback(async () => {
    const [feedResult, meResult, leaderboardResult] = await Promise.allSettled([
      socialAdventureApi.feed(),
      socialAdventureApi.getMine(),
      socialAdventureApi.globalLeaderboard(),
    ]);

    const unavailable: string[] = [];

    if (feedResult.status === 'fulfilled') {
      setPosts(feedResult.value.posts);
      setFeedPrivacy(feedResult.value.privacy);
    } else {
      unavailable.push('feed');
    }

    if (meResult.status === 'fulfilled') {
      setMe(meResult.value);
    } else {
      unavailable.push('your Social Adventure status');
    }

    if (leaderboardResult.status === 'fulfilled') {
      setLeaderboard(leaderboardResult.value);
    } else {
      unavailable.push('global league');
    }

    if (unavailable.length === 0) {
      setError(null);
    } else if (unavailable.length === 3) {
      setError(
        'Social Adventure could not refresh. Previously loaded server content is still shown where available; Woof did not estimate missing score, rank, reaction, or privacy state.'
      );
    } else {
      setError(
        `Community partially refreshed. ${unavailable.join(' and ')} could not be refreshed; previously loaded server content is still shown where available.`
      );
    }

    setLoading(false);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    void loadCommunity();
  }, [loadCommunity]);

  const toggleGlobalVisibility = async () => {
    if (!me || preferenceSaving) return;
    const next = !me.preferences.globalLeaderboardOptIn;
    setPreferenceSaving(true);
    setError(null);
    try {
      const preferences = await socialAdventureApi.updatePreferences(next);
      setMe((current) => (current ? { ...current, preferences } : current));

      const [meResult, leaderboardResult] = await Promise.allSettled([
        socialAdventureApi.getMine(),
        socialAdventureApi.globalLeaderboard(),
      ]);
      const unavailable: string[] = [];

      if (meResult.status === 'fulfilled') {
        setMe(meResult.value);
      } else {
        unavailable.push('your score');
      }

      if (leaderboardResult.status === 'fulfilled') {
        setLeaderboard(leaderboardResult.value);
      } else {
        unavailable.push('the global league');
      }

      if (unavailable.length > 0) {
        setError(
          `Your visibility preference was saved by the server, but ${unavailable.join(' and ')} could not refresh yet.`
        );
      }
    } catch {
      setError(
        'Woof could not change your global league visibility. The last server-confirmed setting is still shown.'
      );
    } finally {
      setPreferenceSaving(false);
    }
  };

  const handleReaction = async (
    shareId: string,
    reaction: SocialAdventureReaction,
    remove: boolean
  ) => {
    const actionKey = `${shareId}:${reaction}`;
    if (reactionSaving) return;
    setReactionSaving(actionKey);
    setError(null);
    try {
      if (remove) await socialAdventureApi.removeReaction(shareId, reaction);
      else await socialAdventureApi.addReaction(shareId, reaction);
      const feedResponse = await socialAdventureApi.feed();
      setPosts(feedResponse.posts);
      setFeedPrivacy(feedResponse.privacy);
    } catch {
      setError(
        'That reaction could not be saved. No league score was changed by the failed action.'
      );
    } finally {
      setReactionSaving(null);
    }
  };

  if (loading) {
    return (
      <View style={styles.loading} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening Social Adventure…</Text>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <SocialAdventureCommunityView
        posts={posts}
        me={me}
        leaderboard={leaderboard}
        feedPrivacy={feedPrivacy}
        error={error}
        refreshing={refreshing}
        preferenceSaving={preferenceSaving}
        reactionSaving={reactionSaving}
        onRefresh={() => {
          setRefreshing(true);
          void loadCommunity();
        }}
        onToggleGlobalVisibility={() => void toggleGlobalVisibility()}
        onReaction={(shareId, reaction, remove) => void handleReaction(shareId, reaction, remove)}
        onOpenExpedition={() => navigation.navigate('Expedition')}
        onOpenSkillcraft={() => navigation.navigate('Skillcraft')}
        onOpenPacks={() => navigation.navigate('Packs')}
        onOpenEvents={() => navigation.navigate('Events')}
        onOpenMap={() => navigation.navigate('Map')}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.background.paper },
  loading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
});
