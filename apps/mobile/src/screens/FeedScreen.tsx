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
    try {
      const [feedResponse, meResponse, leaderboardResponse] = await Promise.all([
        socialAdventureApi.feed(),
        socialAdventureApi.getMine(),
        socialAdventureApi.globalLeaderboard(),
      ]);
      setPosts(feedResponse.posts);
      setFeedPrivacy(feedResponse.privacy);
      setMe(meResponse);
      setLeaderboard(leaderboardResponse);
      setError(null);
    } catch {
      setPosts([]);
      setFeedPrivacy(null);
      setMe(null);
      setLeaderboard(null);
      setError(
        'Social Adventure authority is unavailable right now. Woof did not estimate a score, rank, reaction, or privacy state.'
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
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
      await socialAdventureApi.updatePreferences(next);
      const [meResponse, leaderboardResponse] = await Promise.all([
        socialAdventureApi.getMine(),
        socialAdventureApi.globalLeaderboard(),
      ]);
      setMe(meResponse);
      setLeaderboard(leaderboardResponse);
    } catch {
      setError(
        'Woof could not change your global league visibility. Your prior setting remains authoritative.'
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
