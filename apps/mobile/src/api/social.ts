import apiClient from './client';
import { Comment, CreatePostDto, Post } from '../types';

interface PostEnvelope {
  posts: Post[];
  total: number;
  skip: number;
  take: number;
}

export const socialApi = {
  async getFeed(skip = 0, take = 20): Promise<PostEnvelope> {
    return apiClient.get('/social/posts', { params: { skip, take } });
  },

  async getPost(id: string): Promise<Post> {
    return apiClient.get(`/social/posts/${id}`);
  },

  async createPost(data: CreatePostDto): Promise<Post> {
    return apiClient.post('/social/posts', data);
  },

  async updatePost(
    id: string,
    data: Partial<CreatePostDto> & { mediaUrls?: string[]; visibility?: 'PUBLIC' | 'FRIENDS_ONLY' | 'PRIVATE' },
  ): Promise<Post> {
    return apiClient.put(`/social/posts/${id}`, data);
  },

  async deletePost(id: string): Promise<void> {
    await apiClient.delete(`/social/posts/${id}`);
  },

  async likePost(postId: string): Promise<void> {
    await apiClient.post(`/social/posts/${postId}/likes`);
  },

  async unlikePost(postId: string): Promise<void> {
    await apiClient.delete(`/social/posts/${postId}/likes`);
  },

  async getComments(postId: string): Promise<Comment[]> {
    return apiClient.get(`/social/posts/${postId}/comments`);
  },

  async addComment(postId: string, text: string): Promise<Comment> {
    return apiClient.post(`/social/posts/${postId}/comments`, { text });
  },

  async updateComment(commentId: string, text: string): Promise<Comment> {
    return apiClient.put(`/social/comments/${commentId}`, { text });
  },

  async deleteComment(commentId: string): Promise<void> {
    await apiClient.delete(`/social/comments/${commentId}`);
  },
};
