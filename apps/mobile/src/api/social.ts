import apiClient from './client';
import { Post, Comment, CreatePostDto, PaginatedResponse } from '../types';

export const socialApi = {
  async getFeed(page: number = 1, limit: number = 20): Promise<PaginatedResponse<Post>> {
    return apiClient.get('/social/feed', { params: { page, limit } });
  },

  async getPostById(id: string): Promise<Post> {
    return apiClient.get(`/social/posts/${id}`);
  },

  async createPost(data: CreatePostDto): Promise<Post> {
    return apiClient.post('/social/posts', data);
  },

  async deletePost(id: string): Promise<void> {
    return apiClient.delete(`/social/posts/${id}`);
  },

  async likePost(id: string): Promise<void> {
    return apiClient.post(`/social/posts/${id}/like`);
  },

  async unlikePost(id: string): Promise<void> {
    return apiClient.delete(`/social/posts/${id}/like`);
  },

  async getPostComments(postId: string): Promise<Comment[]> {
    return apiClient.get(`/social/posts/${postId}/comments`);
  },

  async createComment(postId: string, content: string): Promise<Comment> {
    return apiClient.post(`/social/posts/${postId}/comments`, { content });
  },

  async deleteComment(postId: string, commentId: string): Promise<void> {
    return apiClient.delete(`/social/posts/${postId}/comments/${commentId}`);
  },

  async uploadPostMedia(postId: string, mediaUris: string[]): Promise<{ urls: string[] }> {
    const formData = new FormData();
    mediaUris.forEach((uri, index) => {
      formData.append('files', {
        uri,
        type: 'image/jpeg',
        name: `post-media-${index}.jpg`,
      } as any);
    });

    return apiClient.post(`/social/posts/${postId}/media`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
};
