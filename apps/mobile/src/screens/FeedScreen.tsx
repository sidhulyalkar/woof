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
import { socialApi } from '../api/social';
import type { Post } from '../types';
import { colors } from '../theme/tokens';
import type { MainTabParamList, RootStackParamList } from '../navigation/AppNavigator';

type Props = CompositeScreenProps<
  BottomTabScreenProps<MainTabParamList, 'Community'>,
  StackScreenProps<RootStackParamList>
>;

export default function FeedScreen({ navigation }: Props) {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string | null>(null);

  const loadFeed = useCallback(async (targetPage: number) => {
    try {
      const response = await socialApi.getFeed(targetPage, 20);
      setPosts(response.posts);
      setError(null);
    } catch {
      setError('Community is unavailable right now. Woof did not change any social state.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadFeed(page);
  }, [loadFeed, page]);

  const handleRefresh = () => {
    setRefreshing(true);
    if (page === 1) void loadFeed(1);
    else setPage(1);
  };

  const handleLike = async (postId: string) => {
    const post = posts.find((candidate) => candidate.id === postId);
    if (!post) return;

    try {
      if (post.isLiked) {
        await socialApi.unlikePost(postId);
        setPosts((previous) =>
          previous.map((candidate) =>
            candidate.id === postId
              ? {
                  ...candidate,
                  isLiked: false,
                  likesCount: Math.max(0, candidate.likesCount - 1),
                }
              : candidate
          )
        );
      } else {
        await socialApi.likePost(postId);
        setPosts((previous) =>
          previous.map((candidate) =>
            candidate.id === postId
              ? { ...candidate, isLiked: true, likesCount: candidate.likesCount + 1 }
              : candidate
          )
        );
      }
    } catch {
      setError('That reaction could not be saved.');
    }
  };

  const renderPost = ({ item }: { item: Post }) => (
    <View style={styles.postCard}>
      <View style={styles.postHeader}>
        {item.user?.avatarUrl ? (
          <Image source={{ uri: item.user.avatarUrl }} style={styles.avatar} />
        ) : (
          <View style={styles.avatarFallback}>
            <Ionicons name="person-outline" size={19} color={colors.gray[600]} />
          </View>
        )}
        <View style={styles.postHeaderInfo}>
          <Text style={styles.displayName}>
            {item.user?.displayName || item.user?.handle || 'Woof member'}
          </Text>
          {item.user?.handle && <Text style={styles.handle}>@{item.user.handle}</Text>}
        </View>
      </View>

      <Text style={styles.postContent}>{item.content}</Text>

      {item.mediaUrls && item.mediaUrls.length > 0 && (
        <Image source={{ uri: item.mediaUrls[0] }} style={styles.postImage} resizeMode="cover" />
      )}

      <View style={styles.postActions}>
        <Pressable
          accessibilityRole="button"
          accessibilityLabel={item.isLiked ? 'Remove reaction' : 'Cheer this post'}
          style={styles.actionButton}
          onPress={() => void handleLike(item.id)}
        >
          <Ionicons
            name={item.isLiked ? 'heart' : 'heart-outline'}
            size={22}
            color={item.isLiked ? colors.error.main : colors.gray[600]}
          />
          <Text style={styles.actionText}>{item.likesCount}</Text>
        </Pressable>
        <View style={styles.contextOnly}>
          <Ionicons name="chatbubble-outline" size={20} color={colors.gray[400]} />
          <Text style={styles.contextOnlyText}>{item.commentsCount}</Text>
        </View>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color={colors.primary[600]} />
        <Text style={styles.loadingText}>Loading Community…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={posts}
        renderItem={renderPost}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={handleRefresh} />}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.eyebrow}>PEOPLE AROUND YOUR DOG LIFE</Text>
            <Text style={styles.headerTitle}>Community</Text>
            <Text style={styles.headerSubtitle}>
              Real friends, local plans, and shared moments. Community should help you get back to
              life together, not keep you scrolling.
            </Text>
            <View style={styles.quickLinks}>
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
            <Ionicons name="people-outline" size={44} color={colors.primary[500]} />
            <Text style={styles.emptyText}>A quieter community is okay.</Text>
            <Text style={styles.emptySubtext}>
              Woof can still be useful through Today, Story, and your real relationship even when
              there is nothing new to browse.
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
  header: { padding: 18, paddingBottom: 10 },
  eyebrow: { color: colors.text.secondary, fontSize: 10, fontWeight: '700', letterSpacing: 1.4 },
  headerTitle: { marginTop: 3, fontSize: 32, fontWeight: '800', color: colors.text.primary },
  headerSubtitle: { marginTop: 7, fontSize: 14, lineHeight: 20, color: colors.text.secondary },
  quickLinks: { marginTop: 14, flexDirection: 'row', gap: 8 },
  quickLink: {
    minHeight: 42,
    paddingHorizontal: 13,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[100],
  },
  quickLinkText: { color: colors.primary[800], fontSize: 13, fontWeight: '700' },
  errorCard: {
    marginTop: 12,
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
  displayName: { fontSize: 15, fontWeight: '700', color: colors.text.primary },
  handle: { marginTop: 1, fontSize: 12, color: colors.text.secondary },
  postContent: { fontSize: 15, color: colors.gray[800], lineHeight: 22, marginBottom: 12 },
  postImage: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    backgroundColor: colors.gray[200],
    marginBottom: 12,
  },
  postActions: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingTop: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.gray[200],
  },
  actionButton: {
    minHeight: 40,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    paddingRight: 18,
  },
  actionText: { fontSize: 13, color: colors.text.secondary },
  contextOnly: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  contextOnlyText: { fontSize: 13, color: colors.gray[400] },
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
