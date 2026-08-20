import type { Post } from '../types';
import { apiClient } from './client';

type RawPost = {
  id: string;
  text?: string | null;
  mediaUrls?: string[];
  createdAt: string;
  author: {
    id: string;
    handle: string;
    avatarUrl?: string | null;
  };
  pet?: {
    id: string;
    name: string;
    avatarUrl?: string | null;
  } | null;
  likes?: Array<{ id: string }>;
  _count?: {
    likes?: number;
    comments?: number;
  };
};

type FeedEnvelope = {
  posts: RawPost[];
  total: number;
  skip: number;
  take: number;
};

const isVideoUrl = (url: string) => /\.(mp4|mov|webm|m4v)(\?|$)/i.test(url);

const normalizePost = (post: RawPost): Post => {
  const mediaUrl = post.mediaUrls?.[0] ?? '';

  return {
    id: post.id,
    userId: post.author.id,
    userName: post.author.handle,
    userAvatar: post.author.avatarUrl ?? '',
    petName: post.pet?.name ?? 'Woof',
    petAvatar: post.pet?.avatarUrl ?? '',
    mediaUrl,
    mediaType: mediaUrl && isVideoUrl(mediaUrl) ? 'video' : 'image',
    caption: post.text ?? '',
    timestamp: post.createdAt,
    likes: post._count?.likes ?? 0,
    comments: post._count?.comments ?? 0,
    isLiked: Boolean(post.likes?.length),
  };
};

export const webSocialApi = {
  async getFeed(): Promise<Post[]> {
    const response = await apiClient.get('/social/posts') as FeedEnvelope;
    return (response.posts ?? []).map(normalizePost);
  },

  createPost: (data: { text?: string; mediaUrls?: string[]; petId?: string }) =>
    apiClient.post('/social/posts', data),

  likePost: (postId: string) => apiClient.post(`/social/posts/${postId}/likes`),
  unlikePost: (postId: string) => apiClient.delete(`/social/posts/${postId}/likes`),

  addComment: (postId: string, text: string) =>
    apiClient.post(`/social/posts/${postId}/comments`, { text }),

  getComments: (postId: string) => apiClient.get(`/social/posts/${postId}/comments`),
};
