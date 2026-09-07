import React from 'react';
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
import type {
  GlobalLeaderboard,
  SocialAdventureMe,
  SocialAdventurePost,
  SocialAdventureReaction,
} from '../../api/social-adventure';
import { colors } from '../../theme/tokens';

const reactionCopy: Record<SocialAdventureReaction, string> = {
  NICE_READ: 'Nice read',
  GOOD_CALL: 'Good call',
  TRYING_THIS: 'Trying this',
  ADVENTURE_INSPIRATION: 'Adventure inspiration',
  CHEER: 'Cheer',
};

type Props = {
  posts: SocialAdventurePost[];
  me: SocialAdventureMe | null;
  leaderboard: GlobalLeaderboard | null;
  feedPrivacy: string | null;
  error: string | null;
  refreshing: boolean;
  preferenceSaving: boolean;
  reactionSaving: string | null;
  onRefresh: () => void;
  onToggleGlobalVisibility: () => void;
  onReaction: (shareId: string, reaction: SocialAdventureReaction, remove: boolean) => void;
  onOpenSkillcraft: () => void;
  onOpenPacks: () => void;
  onOpenEvents: () => void;
  onOpenMap: () => void;
};

function QuickLink({
  icon,
  label,
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable accessibilityRole="button" style={styles.quickLink} onPress={onPress}>
      <Ionicons name={icon} size={18} color={colors.primary[700]} />
      <Text style={styles.quickLinkText}>{label}</Text>
    </Pressable>
  );
}

function League({
  me,
  leaderboard,
  saving,
  onToggle,
}: {
  me: SocialAdventureMe;
  leaderboard: GlobalLeaderboard;
  saving: boolean;
  onToggle: () => void;
}) {
  const isPublic = me.preferences.globalLeaderboardOptIn;

  return (
    <View style={styles.sectionCard}>
      <View style={styles.headingRow}>
        <View style={styles.flexCopy}>
          <Text style={styles.eyebrow}>OPT-IN LEAGUE</Text>
          <Text style={styles.sectionTitle}>Global human-side league</Text>
        </View>
        <Ionicons name="globe-outline" size={22} color={colors.primary[700]} />
      </View>

      <View style={styles.privacyCard}>
        <Ionicons name="shield-checkmark-outline" size={20} color={colors.success.dark} />
        <View style={styles.flexCopy}>
          <Text style={styles.privacyTitle}>
            {isPublic ? 'You are visible in the global league.' : 'Your score is private by default.'}
          </Text>
          <Text style={styles.privacyText}>
            Opting in publishes your handle and Social Adventure score. Pet health, Daily Signals,
            route data, private notes, and practice-score magnitude stay out.
          </Text>
          {isPublic && leaderboard.me.rank !== null && (
            <Text style={styles.myRank}>Your server-issued rank: #{leaderboard.me.rank}</Text>
          )}
        </View>
      </View>

      <Pressable
        accessibilityRole="button"
        disabled={saving}
        style={[styles.primaryAction, isPublic && styles.outlineAction, saving && styles.disabled]}
        onPress={onToggle}
      >
        {saving ? (
          <ActivityIndicator color={isPublic ? colors.primary[700] : '#ffffff'} />
        ) : (
          <Text style={[styles.primaryActionText, isPublic && styles.outlineActionText]}>
            {isPublic ? 'Make my rank private' : 'Join global league'}
          </Text>
        )}
      </Pressable>

      {leaderboard.entries.length === 0 ? (
        <View style={styles.quietCard}>
          <Text style={styles.quietTitle}>An empty podium is allowed.</Text>
          <Text style={styles.quietText}>
            Private-by-default means Woof does not need to manufacture a leaderboard.
          </Text>
        </View>
      ) : (
        <View style={styles.rows}>
          {leaderboard.entries.slice(0, 5).map((entry) => (
            <View key={entry.userId} style={styles.rankRow}>
              <View style={styles.rankBadge}>
                <Text style={styles.rankText}>{entry.rank}</Text>
              </View>
              <View style={styles.flexCopy}>
                <Text style={styles.handle}>@{entry.handle}</Text>
                <Text style={styles.meta}>
                  {entry.components.humanSkill.score} skill ·{' '}
                  {entry.components.adventureVariety.pathways.length} pathways
                </Text>
              </View>
              <Text style={styles.score}>{entry.score}</Text>
            </View>
          ))}
        </View>
      )}
      <Text style={styles.disclaimer}>{leaderboard.disclaimer}</Text>
    </View>
  );
}

function PostCard({
  post,
  reactionSaving,
  onReaction,
}: {
  post: SocialAdventurePost;
  reactionSaving: string | null;
  onReaction: Props['onReaction'];
}) {
  return (
    <View style={styles.postCard}>
      <View style={styles.postHeader}>
        {post.avatarUrl ? (
          <Image source={{ uri: post.avatarUrl }} style={styles.avatar} />
        ) : (
          <View style={styles.avatarFallback}>
            <Ionicons name="person-outline" size={19} color={colors.gray[600]} />
          </View>
        )}
        <View style={styles.flexCopy}>
          <Text style={styles.handle}>@{post.handle}</Text>
          <Text style={styles.meta}>
            {post.petName ? `with ${post.petName} · ` : ''}{post.kind.replace(/_/g, ' ')}
          </Text>
        </View>
      </View>

      <Text style={styles.postTitle}>{post.headline}</Text>
      <Text style={styles.postSummary}>{post.summary}</Text>
      {post.caption && post.caption !== post.summary && (
        <Text style={styles.caption}>{post.caption}</Text>
      )}

      <View style={styles.reactions}>
        {post.reactions.map((reaction) => {
          const busy = reactionSaving === `${post.shareId}:${reaction.reaction}`;
          return (
            <Pressable
              key={reaction.reaction}
              accessibilityRole="button"
              accessibilityLabel={`${reaction.mine ? 'Remove' : 'Add'} ${reactionCopy[reaction.reaction]} reaction`}
              disabled={Boolean(reactionSaving)}
              style={[styles.reactionChip, reaction.mine && styles.reactionChipMine]}
              onPress={() => onReaction(post.shareId, reaction.reaction, reaction.mine)}
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
      <Text style={styles.boundaryText}>Reactions build culture, not rank.</Text>
    </View>
  );
}

export function SocialAdventureCommunityView(props: Props) {
  const { me, leaderboard } = props;

  return (
    <FlatList
      data={props.posts}
      keyExtractor={(item) => item.shareId}
      contentContainerStyle={styles.listContent}
      refreshControl={<RefreshControl refreshing={props.refreshing} onRefresh={props.onRefresh} />}
      renderItem={({ item }) => (
        <PostCard
          post={item}
          reactionSaving={props.reactionSaving}
          onReaction={props.onReaction}
        />
      )}
      ListHeaderComponent={
        <View style={styles.header}>
          <View style={styles.heroCard}>
            <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
            <Text style={styles.heroTitle}>You compete. Your dog does not.</Text>
            <Text style={styles.bodyText}>
              Human Skill breadth and varied, suitable Adventures can count. Distance, repetition,
              likes, health, symptoms, exercise intensity, and missed days are worth zero league
              points.
            </Text>

            {me && (
              <View style={styles.scoreCard}>
                <View style={styles.headingRow}>
                  <View>
                    <Text style={styles.meta}>This week</Text>
                    <Text style={styles.heroScore}>
                      {me.score}
                      <Text style={styles.scoreMax}> / {me.maxScore}</Text>
                    </Text>
                  </View>
                  <Ionicons name="trophy-outline" size={25} color={colors.primary[700]} />
                </View>
                <Text style={styles.meta}>
                  {me.components.humanSkill.score}/{me.components.humanSkill.maxScore} Human Skill ·{' '}
                  {me.components.adventureVariety.pathways.length} Adventure pathways
                </Text>
              </View>
            )}

            <View style={styles.quickLinks}>
              <QuickLink icon="game-controller-outline" label="Skillcraft" onPress={props.onOpenSkillcraft} />
              <QuickLink icon="people-outline" label="Packs" onPress={props.onOpenPacks} />
              <QuickLink icon="calendar-outline" label="Events" onPress={props.onOpenEvents} />
              <QuickLink icon="map-outline" label="Nearby" onPress={props.onOpenMap} />
            </View>
          </View>

          {me && leaderboard && (
            <League
              me={me}
              leaderboard={leaderboard}
              saving={props.preferenceSaving}
              onToggle={props.onToggleGlobalVisibility}
            />
          )}

          <View style={styles.feedHeading}>
            <Text style={styles.eyebrow}>OPTIONAL SHARING</Text>
            <Text style={styles.sectionTitle}>Adventure feed</Text>
            <Text style={styles.bodyText}>
              Nothing posts automatically. Shared cards are server-authored summaries, and their
              reactions never become pet labels or league points.
            </Text>
            {props.feedPrivacy && <Text style={styles.disclaimer}>{props.feedPrivacy}</Text>}
          </View>

          {props.error && (
            <View style={styles.errorCard} accessibilityRole="alert">
              <Ionicons name="alert-circle-outline" size={18} color={colors.error.dark} />
              <Text style={styles.errorText}>{props.error}</Text>
            </View>
          )}
        </View>
      }
      ListEmptyComponent={
        <View style={styles.emptyCard}>
          <Ionicons name="paw-outline" size={42} color={colors.primary[500]} />
          <Text style={styles.quietTitle}>A quieter community is okay.</Text>
          <Text style={styles.quietText}>
            Nothing needs to be posted for Woof to work. Skillcraft, real-world Adventures, Story,
            and ordinary time together remain the point.
          </Text>
        </View>
      }
    />
  );
}

const styles = StyleSheet.create({
  listContent: { paddingBottom: 110 },
  header: { padding: 14, paddingBottom: 8 },
  heroCard: {
    padding: 18,
    borderRadius: 24,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  heroTitle: { marginTop: 5, color: colors.gray[900], fontSize: 28, lineHeight: 34, fontWeight: '800' },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  bodyText: { marginTop: 7, color: colors.gray[600], fontSize: 13, lineHeight: 20 },
  scoreCard: { marginTop: 15, padding: 14, borderRadius: 16, backgroundColor: '#ffffff' },
  headingRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  heroScore: { marginTop: 2, color: colors.primary[700], fontSize: 27, fontWeight: '900' },
  scoreMax: { color: colors.gray[500], fontSize: 13, fontWeight: '700' },
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
  sectionCard: { marginTop: 14, padding: 17, borderRadius: 22, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff' },
  flexCopy: { flex: 1 },
  privacyCard: { marginTop: 13, padding: 13, borderRadius: 15, backgroundColor: colors.success.light, flexDirection: 'row', gap: 9 },
  privacyTitle: { color: colors.success.dark, fontSize: 13, fontWeight: '800' },
  privacyText: { marginTop: 4, color: colors.success.dark, fontSize: 11, lineHeight: 17 },
  myRank: { marginTop: 7, color: colors.success.dark, fontSize: 11, fontWeight: '800' },
  primaryAction: { minHeight: 45, marginTop: 12, borderRadius: 12, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary[600] },
  outlineAction: { backgroundColor: '#ffffff', borderWidth: 1, borderColor: colors.primary[300] },
  primaryActionText: { color: '#ffffff', fontSize: 13, fontWeight: '800' },
  outlineActionText: { color: colors.primary[800] },
  disabled: { opacity: 0.55 },
  quietCard: { marginTop: 13, padding: 13, borderRadius: 14, backgroundColor: colors.gray[50] },
  quietTitle: { marginTop: 4, color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  quietText: { marginTop: 4, color: colors.gray[600], fontSize: 12, lineHeight: 18 },
  rows: { marginTop: 13, gap: 7 },
  rankRow: { minHeight: 58, padding: 9, borderRadius: 13, flexDirection: 'row', alignItems: 'center', backgroundColor: colors.gray[50] },
  rankBadge: { width: 34, height: 34, borderRadius: 11, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.primary[100] },
  rankText: { color: colors.primary[800], fontSize: 12, fontWeight: '900' },
  handle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  meta: { marginTop: 2, color: colors.gray[600], fontSize: 10 },
  score: { color: colors.primary[700], fontSize: 17, fontWeight: '900' },
  disclaimer: { marginTop: 9, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
  feedHeading: { paddingHorizontal: 4, paddingTop: 25, paddingBottom: 11 },
  errorCard: { marginTop: 10, padding: 12, borderRadius: 12, flexDirection: 'row', gap: 7, backgroundColor: colors.error.light },
  errorText: { flex: 1, color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  postCard: { marginHorizontal: 12, marginBottom: 10, padding: 16, borderRadius: 18, borderWidth: 1, borderColor: colors.gray[200], backgroundColor: '#ffffff' },
  postHeader: { flexDirection: 'row', alignItems: 'center', gap: 11, marginBottom: 11 },
  avatar: { width: 40, height: 40, borderRadius: 20, backgroundColor: colors.gray[200] },
  avatarFallback: { width: 40, height: 40, borderRadius: 20, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.gray[100] },
  postTitle: { color: colors.gray[900], fontSize: 17, fontWeight: '800' },
  postSummary: { marginTop: 6, color: colors.gray[600], fontSize: 13, lineHeight: 20 },
  caption: { marginTop: 10, padding: 11, borderRadius: 12, color: colors.gray[800], backgroundColor: colors.gray[50], fontSize: 12, lineHeight: 18 },
  reactions: { marginTop: 14, flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  reactionChip: { minHeight: 34, paddingHorizontal: 10, borderRadius: 999, borderWidth: 1, borderColor: colors.gray[200], alignItems: 'center', justifyContent: 'center', backgroundColor: colors.gray[50] },
  reactionChipMine: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  reactionText: { color: colors.gray[600], fontSize: 10, fontWeight: '700' },
  reactionTextMine: { color: colors.primary[800] },
  boundaryText: { marginTop: 8, color: colors.gray[400], fontSize: 9 },
  emptyCard: { alignItems: 'center', paddingHorizontal: 38, paddingVertical: 56 },
});