import React, { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Image,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
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
import { colors } from '../theme/tokens';
import type { MainTabParamList, RootStackParamList } from '../navigation/AppNavigator';

type Props = CompositeScreenProps<
  BottomTabScreenProps<MainTabParamList, 'Community'>,
  StackScreenProps<RootStackParamList>
>;

const reactionCopy: Record<SocialAdventureReaction, string> = {
  NICE_READ: 'Nice read',
  GOOD_CALL: 'Good call',
  TRYING_THIS: 'Trying this',
  ADVENTURE_INSPIRATION: 'Adventure inspiration',
  CHEER: 'Cheer',
};

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

  const handleRefresh = () => {
    setRefreshing(true);
    void loadCommunity();
  };

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
      setError('Woof could not change your global league visibility. Your prior setting remains authoritative.');
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
      setError('That reaction could not be saved. No league score was changed by the failed action.');
    } finally {
      setReactionSaving(null);
    }
  };

  const renderPost = ({ item }: { item: SocialAdventurePost }) => (
    <View style={styles.postCard}>
      <View style={styles.postHeader}>
        {item.avatarUrl ? (
          <Image source={{ uri: item.avatarUrl }} style={styles.avatar} />
        ) : (
          <View style={styles.avatarFallback}>
            <Ionicons name="person-outline" size={19} color={colors.gray[600]} />
          </View>
        )}
        <View style={styles.postHeaderInfo}>
          <View style={styles.authorLine}>
            <Text style={styles.handle}>@{item.handle}</Text>
            {item.petName && <Text style={styles.petContext}>with {item.petName}</Text>}
          </View>
          <Text style={styles.kind}>{item.kind.replace(/_/g, ' ')}</Text>
        </View>
      </View>

      <Text style={styles.postTitle}>{item.headline}</Text>
      <Text style={styles.postSummary}>{item.summary}</Text>
      {item.caption && item.caption !== item.summary && (
        <View style={styles.captionCard}>
          <Text style={styles.captionText}>{item.caption}</Text>
        </View>
      )}

      <View style={styles.reactions}>
        {item.reactions.map((reaction) => {
          const busy = reactionSaving === `${item.shareId}:${reaction.reaction}`;
          return (
            <Pressable
              key={reaction.reaction}
              accessibilityRole="button"
              accessibilityLabel={`${reaction.mine ? 'Remove' : 'Add'} ${reactionCopy[
                reaction.reaction
              ]} reaction`}
              disabled={Boolean(reactionSaving)}
              style={[styles.reactionChip, reaction.mine && styles.reactionChipMine]}
              onPress={() =>
                void handleReaction(item.shareId, reaction.reaction, reaction.mine)
              }
            >
              {busy ? (
                <ActivityIndicator size="small" color={colors.primary[700]} />
              ) : (
                <Text style={[styles.reactionText, reaction.mine && styles.reactionTextMine]}>
                  {reactionCopy[reaction.reaction]}
                  {reaction.count > 0 ? ` · ${reaction.count}` : ''}
                </Text>
              )}
            </Pressable>
          );
        })}
      </View>
      <Text style={styles.reactionBoundary}>Reactions build culture, not rank.</Text>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centerContainer} accessibilityRole="progressbar">
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Opening Social Adventure…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={posts}
        renderItem={renderPost}
        keyExtractor={(item) => item.shareId}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />}
        ListHeaderComponent={
          <View style={styles.header}>
            <View style={styles.heroCard}>
              <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
              <Text style={styles.headerTitle}>You compete. Your dog doesn't.</Text>
              <Text style={styles.headerSubtitle}>
                Human Skill breadth and varied, suitable Adventures can count. Distance, repetition,
                likes, health, symptoms, exercise intensity, and missed days are worth zero league
                points.
              </Text>

              {me && (
                <View style={styles.scoreCard}>
                  <View style={styles.scoreTopline}>
                    <View>
                      <Text style={styles.scoreLabel}>This week</Text>
                      <Text style={styles.scoreValue}>
                        {me.score}
                        <Text style={styles.scoreMax}> / {me.maxScore}</Text>
                      </Text>
                    </View>
                    <Ionicons name="trophy-outline" size={26} color={colors.primary[700]} />
                  </View>
                  <View style={styles.scoreComponents}>
                    <View style={styles.componentCard}>
                      <Text style={styles.componentTitle}>Human Skill</Text>
                      <Text style={styles.componentValue}>
                        {me.components.humanSkill.score}/{me.components.humanSkill.maxScore}
                      </Text>
                    </View>
                    <View style={styles.componentCard}>
                      <Text style={styles.componentTitle}>Adventure variety</Text>
                      <Text style={styles.componentValue}>
                        {me.components.adventureVariety.pathways.length} pathways
                      </Text>
                    </View>
                  </View>
                </View>
              )}

              <View style={styles.quickLinks}>
                <Pressable
                  style={styles.quickLink}
                  onPress={() => navigation.navigate('Skillcraft')}
                  accessibilityRole="button"
                >
                  <Ionicons name="game-controller-outline" size={18} color={colors.primary[700]} />
                  <Text style={styles.quickLinkText}>Skillcraft</Text>
                </Pressable>
                <Pressable
                  style={styles.quickLink}
                  onPress={() => navigation.navigate('Packs')}
                  accessibilityRole="button"
                >
                  <Ionicons name="people-outline" size={18} color={colors.primary[700]} />
                  <Text style={styles.quickLinkText}>Packs</Text>
                </Pressable>
                <Pressable
                  style={styles.quickLink}
                  onPress={() => navigation.navigate('Events')}
                  accessibilityRole="button"
                >
                  <Ionicons name="calendar-outline" size={18} color={colors.primary[700]} />
                  <Text style={styles.quickLinkText}>Events</Text>
                </Pressable>
                <Pressable
                  style={styles.quickLink}
                  onPress={() => navigation.navigate('Map')}
                  accessibilityRole="button"
                >
                  <Ionicons name="map-outline" size={18} color={colors.primary[700]} />
                  <Text style={styles.quickLinkText}>Nearby</Text>
                </Pressable>
              </View>
            </View>

            {me && leaderboard && (
              <View style={styles.leagueCard}>
                <View style={styles.leagueHeading}>
                  <View style={styles.leagueHeadingCopy}>
                    <Text style={styles.sectionEyebrow}>OPT-IN LEAGUE</Text>
                    <Text style={styles.sectionTitle}>Global human-side league</Text>
                  </View>
                  <Ionicons name="globe-outline" size={22} color={colors.primary[700]} />
                </View>

                <View style={styles.privacyCard}>
                  <Ionicons name="shield-checkmark-outline" size={20} color={colors.success.dark} />
                  <View style={styles.privacyCopy}>
                    <Text style={styles.privacyTitle}>
                      {me.preferences.globalLeaderboardOptIn
                        ? 'You are visible in the global league.'
                        : 'Your score is private by default.'}
                    </Text>
                    <Text style={styles.privacyText}>
                      Opting in publishes your handle and Social Adventure score. It does not publish
                      pet health, Daily Signals, route data, private notes, or practice-score magnitude.
                    </Text>
                    {me.preferences.globalLeaderboardOptIn && leaderboard.me.rank !== null && (
                      <Text style={styles.myRank}>Your server-issued rank: #{leaderboard.me.rank}</Text>
                    )}
                  </View>
                </View>

                <Pressable
                  accessibilityRole="button"
                  disabled={preferenceSaving}
                  style={[
                    styles.visibilityButton,
                    me.preferences.globalLeaderboardOptIn && styles.visibilityButtonPrivate,
                    preferenceSaving && styles.disabled,
                  ]}
                  onPress={() => void toggleGlobalVisibility()}
                >
                  {preferenceSaving ? (
                    <ActivityIndicator color="#ffffff" />
                  ) : (
                    <Text
                      style={[
                        styles.visibilityButtonText,
                        me.preferences.globalLeaderboardOptIn && styles.visibilityButtonPrivateText,
                      ]}
                    >
                      {me.preferences.globalLeaderboardOptIn
                        ? 'Make my rank private'
                        : 'Join global league'}
                    </Text>
                  )}
                </Pressable>

                {leaderboard.entries.length === 0 ? (
                  <View style={styles.emptyLeague}>
                    <Text style={styles.emptyLeagueTitle}>An empty podium is allowed.</Text>
                    <Text style={styles.emptyLeagueText}>
                      Private-by-default means Woof does not need to manufacture a leaderboard.
                    </Text>
                  </View>
                ) : (
                  <View style={styles.leagueRows}>
                    {leaderboard.entries.slice(0, 5).map((entry) => (
                      <View key={entry.userId} style={styles.leagueRow}>
                        <View style={styles.rankBadge}>
                          <Text style={styles.rankText}>{entry.rank}</Text>
                        </View>
                        <View style={styles.leagueRowCopy}>
                          <Text style={styles.leagueHandle}>@{entry.handle}</Text>
                          <Text style={styles.leagueMeta}>
                            {entry.components.humanSkill.score} skill ·{' '}
                            {entry.components.adventureVariety.pathways.length} pathways
                          </Text>
                        </View>
                        <Text style={styles.leagueScore}>{entry.score}</Text>
                      </View>
                    ))}
                  </View>
                )}
                <Text style={styles.disclaimer}>{leaderboard.disclaimer}</Text>
              </View>
            )}

            <View style={styles.feedHeading}>
              <Text style={styles.sectionEyebrow}>OPTIONAL SHARING</Text>
              <Text style={styles.sectionTitle}>Adventure feed</Text>
              <Text style={styles.feedIntro}>
                Nothing posts automatically. Shared cards are server-authored summaries, and their
                reactions never become pet labels or league points.
              </Text>
              {feedPrivacy && <Text style={styles.feedPrivacy}>{feedPrivacy}</Text>}
            </View>

            {error && (
              <View style={styles.errorCard} accessibilityRole="alert">
                <Ionicons name="alert-circle-outline" size={18} color={colors.error.dark} />
                <Text style={styles.errorText}>{error}</Text>
              </View>
            )}
          </View>
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Ionicons name="paw-outline" size={44} color={colors.primary[500]} />
            <Text style={styles.emptyText}>A quieter community is okay.</Text>
            <Text style={styles.emptySubtext}>
              Nothing needs to be posted for Woof to work. Skillcraft, real-world Adventures, Story,
              and ordinary time together remain the point.
            </Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.background.paper },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.paper,
  },
  loadingText: { marginTop: 12, color: colors.text.secondary },
  listContent: { paddingBottom: 110 },
  header: { padding: 14, paddingBottom: 8 },
  heroCard: {
    padding: 18,
    borderRadius: 24,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.4 },
  headerTitle: {
    marginTop: 5,
    fontSize: 29,
    lineHeight: 35,
    fontWeight: '800',
    color: colors.text.primary,
  },
  headerSubtitle: { marginTop: 8, fontSize: 14, lineHeight: 21, color: colors.text.secondary },
  scoreCard: { marginTop: 17, padding: 15, borderRadius: 17, backgroundColor: '#ffffff' },
  scoreTopline: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  scoreLabel: { color: colors.gray[600], fontSize: 11, fontWeight: '700' },
  scoreValue: { marginTop: 2, color: colors.primary[700], fontSize: 28, fontWeight: '900' },
  scoreMax: { color: colors.gray[500], fontSize: 13, fontWeight: '700' },
  scoreComponents: { marginTop: 11, flexDirection: 'row', gap: 8 },
  componentCard: { flex: 1, padding: 10, borderRadius: 12, backgroundColor: colors.gray[50] },
  componentTitle: { color: colors.gray[800], fontSize: 11, fontWeight: '800' },
  componentValue: { marginTop: 3, color: colors.gray[600], fontSize: 11 },
  quickLinks: { marginTop: 14, flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  quickLink: {
    minHeight: 42,
    paddingHorizontal: 12,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  quickLinkText: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  leagueCard: {
    marginTop: 14,
    padding: 17,
    borderRadius: 22,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: '#ffffff',
  },
  leagueHeading: { flexDirection: 'row', alignItems: 'flex-start', gap: 10 },
  leagueHeadingCopy: { flex: 1 },
  sectionEyebrow: {
    color: colors.primary[700],
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 1.1,
  },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  privacyCard: {
    marginTop: 13,
    padding: 13,
    borderRadius: 15,
    backgroundColor: colors.success.light,
    flexDirection: 'row',
    gap: 9,
  },
  privacyCopy: { flex: 1 },
  privacyTitle: { color: colors.success.dark, fontSize: 13, fontWeight: '800' },
  privacyText: { marginTop: 4, color: colors.success.dark, fontSize: 11, lineHeight: 17 },
  myRank: { marginTop: 7, color: colors.success.dark, fontSize: 11, fontWeight: '800' },
  visibilityButton: {
    minHeight: 45,
    marginTop: 12,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[600],
  },
  visibilityButtonPrivate: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.primary[300],
  },
  visibilityButtonText: { color: '#ffffff', fontSize: 13, fontWeight: '800' },
  visibilityButtonPrivateText: { color: colors.primary[800] },
  disabled: { opacity: 0.55 },
  emptyLeague: { marginTop: 13, padding: 13, borderRadius: 14, backgroundColor: colors.gray[50] },
  emptyLeagueTitle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  emptyLeagueText: { marginTop: 4, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  leagueRows: { marginTop: 13, gap: 7 },
  leagueRow: {
    minHeight: 58,
    padding: 9,
    borderRadius: 13,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.gray[50],
  },
  rankBadge: {
    width: 34,
    height: 34,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[100],
  },
  rankText: { color: colors.primary[800], fontSize: 12, fontWeight: '900' },
  leagueRowCopy: { flex: 1, marginLeft: 10 },
  leagueHandle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  leagueMeta: { marginTop: 2, color: colors.gray[600], fontSize: 10 },
  leagueScore: { color: colors.primary[700], fontSize: 17, fontWeight: '900' },
  disclaimer: { marginTop: 11, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
  feedHeading: { paddingHorizontal: 4, paddingTop: 26, paddingBottom: 12 },
  feedIntro: { marginTop: 5, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  feedPrivacy: { marginTop: 5, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
  errorCard: {
    marginTop: 10,
    padding: 12,
    borderRadius: 12,
    flexDirection: 'row',
    gap: 7,
    backgroundColor: colors.error.light,
  },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  postCard: {
    backgroundColor: '#ffffff',
    marginHorizontal: 12,
    marginBottom: 10,
    padding: 16,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  postHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  avatar: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.gray[200] },
  avatarFallback: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.gray[100],
  },
  postHeaderInfo: { marginLeft: 12, flex: 1 },
  authorLine: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 6 },
  handle: { fontSize: 14, fontWeight: '800', color: colors.text.primary },
  petContext: { fontSize: 11, color: colors.gray[500] },
  kind: {
    marginTop: 3,
    color: colors.primary[700],
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.9,
  },
  postTitle: { fontSize: 17, fontWeight: '800', color: colors.gray[900] },
  postSummary: { marginTop: 6, fontSize: 13, color: colors.gray[600], lineHeight: 20 },
  captionCard: { marginTop: 10, padding: 11, borderRadius: 12, backgroundColor: colors.gray[50] },
  captionText: { color: colors.gray[800], fontSize: 12, lineHeight: 18 },
  reactions: { marginTop: 14, flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  reactionChip: {
    minHeight: 34,
    paddingHorizontal: 10,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[200],
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.gray[50],
  },
  reactionChipMine: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  reactionText: { color: colors.gray[600], fontSize: 10, fontWeight: '700' },
  reactionTextMine: { color: colors.primary[800] },
  reactionBoundary: { marginTop: 8, color: colors.gray[400], fontSize: 9 },
  emptyContainer: { alignItems: 'center', paddingHorizontal: 38, paddingVertical: 56 },
  emptyText: { marginTop: 12, fontSize: 17, fontWeight: '700', color: colors.text.primary },
  emptySubtext: {
    marginTop: 7,
    fontSize: 13,
    lineHeight: 19,
    color: colors.text.secondary,
    textAlign: 'center',
  },
});
