import React from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
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

function ActionChip({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={styles.actionChip}>
      <Text style={styles.actionChipText}>{label}</Text>
    </Pressable>
  );
}

function LeagueCard({
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
    <View style={styles.card}>
      <Text style={styles.eyebrow}>OPTIONAL LEAGUE</Text>
      <Text style={styles.sectionTitle}>Global human-side league</Text>
      <Text style={styles.body}>
        {isPublic ? 'You are visible in the global league.' : 'Your score is private by default.'}
      </Text>
      <Text style={styles.smallCopy}>
        Opting in publishes your handle and Social Adventure score. Pet health, route data, private
        notes, and practice-score magnitude stay out.
      </Text>

      {isPublic && leaderboard.me.rank !== null && (
        <Text style={styles.rankCallout}>Your server-issued rank: #{leaderboard.me.rank}</Text>
      )}

      <Pressable
        accessibilityRole="button"
        disabled={saving}
        onPress={onToggle}
        style={[styles.primaryButton, saving && styles.disabled]}
      >
        {saving ? (
          <ActivityIndicator color="#ffffff" />
        ) : (
          <Text style={styles.primaryButtonText}>
            {isPublic ? 'Make my rank private' : 'Join global league'}
          </Text>
        )}
      </Pressable>

      {leaderboard.entries.length === 0 ? (
        <View style={styles.quietBox}>
          <Text style={styles.quietTitle}>An empty podium is allowed.</Text>
          <Text style={styles.smallCopy}>
            Private-by-default means Woof does not manufacture a leaderboard.
          </Text>
        </View>
      ) : (
        <View style={styles.rankList}>
          {leaderboard.entries.slice(0, 5).map((entry) => (
            <View key={entry.userId} style={styles.rankRow}>
              <Text style={styles.rankNumber}>#{entry.rank}</Text>
              <View style={styles.flex}>
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
      <Text style={styles.handle}>@{post.handle}</Text>
      <Text style={styles.meta}>
        {post.petName ? `with ${post.petName} · ` : ''}
        {post.kind.replace(/_/g, ' ')}
      </Text>
      <Text style={styles.postTitle}>{post.headline}</Text>
      <Text style={styles.body}>{post.summary}</Text>
      {post.caption && post.caption !== post.summary && (
        <Text style={styles.caption}>{post.caption}</Text>
      )}

      <View style={styles.reactionRow}>
        {post.reactions.map((reaction) => {
          const busy = reactionSaving === `${post.shareId}:${reaction.reaction}`;
          return (
            <Pressable
              key={reaction.reaction}
              accessibilityRole="button"
              accessibilityLabel={`${reaction.mine ? 'Remove' : 'Add'} ${
                reactionCopy[reaction.reaction]
              } reaction`}
              disabled={Boolean(reactionSaving)}
              onPress={() => onReaction(post.shareId, reaction.reaction, reaction.mine)}
              style={[styles.reactionChip, reaction.mine && styles.reactionChipMine]}
            >
              {busy ? (
                <ActivityIndicator size="small" color={colors.primary[700]} />
              ) : (
                <Text style={styles.reactionText}>
                  {reactionCopy[reaction.reaction]}
                  {reaction.count > 0 ? ` · ${reaction.count}` : ''}
                </Text>
              )}
            </Pressable>
          );
        })}
      </View>
      <Text style={styles.boundary}>Reactions build culture, not rank.</Text>
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
        <PostCard post={item} reactionSaving={props.reactionSaving} onReaction={props.onReaction} />
      )}
      ListHeaderComponent={
        <View style={styles.header}>
          <View style={styles.hero}>
            <Text style={styles.eyebrow}>SOCIAL ADVENTURE</Text>
            <Text style={styles.heroTitle}>You compete. Your dog does not.</Text>
            <Text style={styles.body}>
              Human Skill breadth and varied, suitable Adventures can count. Distance, repetition,
              likes, health, symptoms, exercise intensity, and missed days are worth zero league
              points.
            </Text>

            {me && (
              <View style={styles.scoreBox}>
                <Text style={styles.meta}>THIS WEEK</Text>
                <Text style={styles.heroScore}>
                  {me.score} / {me.maxScore}
                </Text>
                <Text style={styles.smallCopy}>
                  {me.components.humanSkill.score}/{me.components.humanSkill.maxScore} Human Skill ·{' '}
                  {me.components.adventureVariety.pathways.length} Adventure pathways
                </Text>
              </View>
            )}

            <View style={styles.actionRow}>
              <ActionChip label="Skillcraft" onPress={props.onOpenSkillcraft} />
              <ActionChip label="Packs" onPress={props.onOpenPacks} />
              <ActionChip label="Events" onPress={props.onOpenEvents} />
              <ActionChip label="Nearby" onPress={props.onOpenMap} />
            </View>
          </View>

          {me && leaderboard && (
            <LeagueCard
              me={me}
              leaderboard={leaderboard}
              saving={props.preferenceSaving}
              onToggle={props.onToggleGlobalVisibility}
            />
          )}

          <View style={styles.feedHeading}>
            <Text style={styles.eyebrow}>OPTIONAL SHARING</Text>
            <Text style={styles.sectionTitle}>Adventure feed</Text>
            <Text style={styles.body}>
              Nothing posts automatically. Shared cards are server-authored summaries, and their
              reactions never become pet labels or league points.
            </Text>
            {props.feedPrivacy && <Text style={styles.disclaimer}>{props.feedPrivacy}</Text>}
          </View>

          {props.error && (
            <View style={styles.errorBox} accessibilityRole="alert">
              <Text style={styles.errorText}>{props.error}</Text>
            </View>
          )}
        </View>
      }
      ListEmptyComponent={
        <View style={styles.emptyBox}>
          <Text style={styles.quietTitle}>A quieter community is okay.</Text>
          <Text style={styles.smallCopy}>
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
  header: { padding: 14 },
  hero: {
    padding: 18,
    borderRadius: 22,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  card: {
    marginTop: 14,
    padding: 16,
    borderRadius: 18,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  eyebrow: { color: colors.primary[700], fontSize: 10, fontWeight: '800', letterSpacing: 1.2 },
  heroTitle: {
    marginTop: 5,
    color: colors.gray[900],
    fontSize: 28,
    lineHeight: 34,
    fontWeight: '800',
  },
  sectionTitle: { marginTop: 3, color: colors.gray[900], fontSize: 20, fontWeight: '800' },
  body: { marginTop: 7, color: colors.gray[600], fontSize: 13, lineHeight: 20 },
  smallCopy: { marginTop: 5, color: colors.gray[600], fontSize: 11, lineHeight: 17 },
  scoreBox: { marginTop: 14, padding: 13, borderRadius: 14, backgroundColor: '#ffffff' },
  heroScore: { marginTop: 2, color: colors.primary[700], fontSize: 26, fontWeight: '900' },
  actionRow: { marginTop: 14, flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  actionChip: {
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderRadius: 999,
    backgroundColor: '#ffffff',
  },
  actionChipText: { color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  primaryButton: {
    marginTop: 12,
    padding: 12,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: colors.primary[600],
  },
  primaryButtonText: { color: '#ffffff', fontWeight: '800' },
  disabled: { opacity: 0.55 },
  quietBox: { marginTop: 12, padding: 12, borderRadius: 12, backgroundColor: colors.gray[50] },
  quietTitle: { color: colors.gray[900], fontSize: 14, fontWeight: '800' },
  rankCallout: { marginTop: 8, color: colors.primary[800], fontSize: 12, fontWeight: '800' },
  rankList: { marginTop: 12, gap: 7 },
  rankRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    padding: 9,
    borderRadius: 11,
    backgroundColor: colors.gray[50],
  },
  rankNumber: { width: 32, color: colors.primary[800], fontWeight: '900' },
  flex: { flex: 1 },
  handle: { color: colors.gray[900], fontSize: 13, fontWeight: '800' },
  meta: { marginTop: 2, color: colors.gray[500], fontSize: 10 },
  score: { color: colors.primary[700], fontSize: 16, fontWeight: '900' },
  disclaimer: { marginTop: 8, color: colors.gray[500], fontSize: 10, lineHeight: 15 },
  feedHeading: { paddingHorizontal: 4, paddingTop: 24, paddingBottom: 9 },
  errorBox: { marginTop: 10, padding: 12, borderRadius: 12, backgroundColor: colors.error.light },
  errorText: { color: colors.error.dark, fontSize: 12, lineHeight: 17 },
  postCard: {
    marginHorizontal: 14,
    marginBottom: 10,
    padding: 15,
    borderRadius: 16,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  postTitle: { marginTop: 10, color: colors.gray[900], fontSize: 17, fontWeight: '800' },
  caption: {
    marginTop: 9,
    padding: 10,
    borderRadius: 10,
    color: colors.gray[700],
    backgroundColor: colors.gray[50],
  },
  reactionRow: { marginTop: 12, flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  reactionChip: {
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: colors.gray[200],
    backgroundColor: colors.gray[50],
  },
  reactionChipMine: { borderColor: colors.primary[300], backgroundColor: colors.primary[50] },
  reactionText: { color: colors.gray[700], fontSize: 10, fontWeight: '700' },
  boundary: { marginTop: 8, color: colors.gray[400], fontSize: 9 },
  emptyBox: {
    marginHorizontal: 14,
    padding: 24,
    borderRadius: 16,
    alignItems: 'center',
    backgroundColor: colors.gray[50],
  },
});
