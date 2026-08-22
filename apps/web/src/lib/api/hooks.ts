import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
  type UseQueryOptions,
} from '@tanstack/react-query';
import { apiClient } from './client';
import { useSessionStore } from '@/store/session';

type ManagedQueryOptions<T> = Omit<
  UseQueryOptions<T, Error, T, readonly unknown[]>,
  'queryKey' | 'queryFn'
>;

export interface Post {
  id: string;
  userId: string;
  petId?: string;
  activityId?: string;
  content: string;
  images?: string[];
  location?: {
    latitude: number;
    longitude: number;
    name?: string;
  };
  activityData?: {
    distance?: number;
    duration?: number;
    calories?: number;
  };
  likes: number;
  comments: number;
  createdAt: string;
  user: {
    id: string;
    username: string;
    avatar?: string;
  };
  pet?: {
    id: string;
    name: string;
    avatar?: string;
  };
}

export interface Activity {
  id: string;
  userId: string;
  petId?: string;
  type: string;
  startedAt: string;
  endedAt?: string;
  route?: unknown;
  humanMetrics?: {
    steps?: number;
    calories?: number;
    hr_avg?: number;
  };
  petMetrics?: {
    distance?: string | number;
    active_time?: number;
  };
  jointMetrics?: Record<string, unknown>;
  createdAt: string;
}

export interface LeaderboardEntry {
  rank: number;
  userId: string;
  username: string;
  avatar?: string;
  petName?: string;
  petAvatar?: string;
  distance: number;
  score: number;
}

export interface SuggestedMatch {
  id: string;
  user: {
    id: string;
    handle: string;
    avatarUrl?: string;
    bio?: string;
  };
  pet: {
    id: string;
    name: string;
    breed: string;
    age: number;
    avatarUrl?: string;
  };
  compatibilityScore: number;
  explainability: {
    topReasons: string[];
    proximityKm?: number;
    mutualInterests?: string[];
  };
  distance?: number;
  lastActive?: string;
}

export const queryKeys = {
  feed: ['feed'] as const,
  activities: ['activities'] as const,
  leaderboard: (timeframe: 'weekly' | 'monthly') => ['leaderboard', timeframe] as const,
  userProfile: (userId: string) => ['user', userId] as const,
  petProfile: (petId: string) => ['pet', petId] as const,
  friends: ['friends'] as const,
  messages: ['messages'] as const,
  conversation: (conversationId: string) => ['conversation', conversationId] as const,
};

function transformBackendPost(backendPost: any): Post {
  return {
    id: backendPost.id,
    userId: backendPost.authorUserId,
    petId: backendPost.petId,
    activityId: backendPost.activityId,
    content: backendPost.text || '',
    images: backendPost.mediaUrls || [],
    likes: backendPost._count?.likes || 0,
    comments: backendPost._count?.comments || 0,
    createdAt: backendPost.createdAt,
    user: {
      id: backendPost.author?.id || backendPost.authorUserId,
      username: backendPost.author?.handle || 'Unknown',
      avatar: backendPost.author?.avatarUrl,
    },
    pet: backendPost.pet
      ? {
          id: backendPost.pet.id,
          name: backendPost.pet.name,
          avatar: backendPost.pet.avatarUrl,
        }
      : undefined,
  };
}

export function useFeed(options?: ManagedQueryOptions<Post[]>) {
  return useQuery<Post[]>({
    queryKey: queryKeys.feed,
    queryFn: async () => {
      const response = await apiClient.get<any>('/social/posts?skip=0&take=20');
      const posts = Array.isArray(response) ? response : response.posts || [];
      return posts.map(transformBackendPost);
    },
    ...options,
  });
}

export function usePost(postId: string, options?: ManagedQueryOptions<Post>) {
  return useQuery<Post>({
    queryKey: ['post', postId],
    queryFn: async () => {
      const response = await apiClient.get<any>(`/social/posts/${postId}`);
      return transformBackendPost(response);
    },
    ...options,
  });
}

export function useCreatePost(options?: UseMutationOptions<Post, Error, Partial<Post>>) {
  const queryClient = useQueryClient();

  return useMutation<Post, Error, Partial<Post>>({
    mutationFn: async (data) => {
      const userId = useSessionStore.getState().user?.id;
      if (!userId) throw new Error('User not authenticated');

      const payload: Record<string, unknown> = {
        text: data.content || '',
        mediaUrls: data.images || [],
        author: { connect: { id: userId } },
      };

      if (data.petId) payload.pet = { connect: { id: data.petId } };
      if (data.activityId) payload.activityId = data.activityId;

      const result = await apiClient.post<any>('/social/posts', payload);
      return transformBackendPost(result);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.feed });
    },
    ...options,
  });
}

export function useLikePost(options?: UseMutationOptions<void, Error, string>) {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: (postId) => {
      const userId = useSessionStore.getState().user?.id;
      if (!userId) throw new Error('User not authenticated');
      return apiClient.post<void>(`/social/posts/${postId}/likes`, { userId });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.feed });
    },
    ...options,
  });
}

export interface Comment {
  id: string;
  postId: string;
  authorUserId: string;
  text: string;
  createdAt: string;
  author?: {
    id: string;
    handle: string;
    avatarUrl?: string;
  };
}

export function useComments(postId: string, options?: ManagedQueryOptions<Comment[]>) {
  return useQuery<Comment[]>({
    queryKey: ['comments', postId],
    queryFn: async () => {
      const response = await apiClient.get<any[]>(`/social/posts/${postId}/comments`);
      return response.map((comment) => ({
        id: comment.id,
        postId: comment.postId,
        authorUserId: comment.authorUserId,
        text: comment.text,
        createdAt: comment.createdAt,
        author: comment.author
          ? {
              id: comment.author.id,
              handle: comment.author.handle,
              avatarUrl: comment.author.avatarUrl,
            }
          : undefined,
      }));
    },
    ...options,
  });
}

export function useCreateComment(
  options?: UseMutationOptions<Comment, Error, { postId: string; text: string }>
) {
  const queryClient = useQueryClient();

  return useMutation<Comment, Error, { postId: string; text: string }>({
    mutationFn: async ({ postId, text }) => {
      const userId = useSessionStore.getState().user?.id;
      if (!userId) throw new Error('User not authenticated');

      const response = await apiClient.post<any>(`/social/posts/${postId}/comments`, {
        text,
        user: { connect: { id: userId } },
      });

      return {
        id: response.id,
        postId: response.postId,
        authorUserId: response.userId,
        text: response.text,
        createdAt: response.createdAt,
        author: response.user
          ? {
              id: response.user.id,
              handle: response.user.handle,
              avatarUrl: response.user.avatarUrl,
            }
          : undefined,
      };
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['comments', variables.postId] });
      queryClient.invalidateQueries({ queryKey: queryKeys.feed });
    },
    ...options,
  });
}

export function useDeleteComment(
  options?: UseMutationOptions<void, Error, { commentId: string; postId: string }>
) {
  const queryClient = useQueryClient();

  return useMutation<void, Error, { commentId: string; postId: string }>({
    mutationFn: ({ commentId }) => apiClient.delete<void>(`/social/comments/${commentId}`),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['comments', variables.postId] });
      queryClient.invalidateQueries({ queryKey: queryKeys.feed });
    },
    ...options,
  });
}

export function useActivities(options?: ManagedQueryOptions<Activity[]>) {
  return useQuery<Activity[]>({
    queryKey: queryKeys.activities,
    queryFn: () => apiClient.get<Activity[]>('/activities'),
    ...options,
  });
}

export function useCreateActivity(
  options?: UseMutationOptions<Activity, Error, Partial<Activity>>
) {
  const queryClient = useQueryClient();

  return useMutation<Activity, Error, Partial<Activity>>({
    mutationFn: (data) => apiClient.post<Activity>('/activities', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.activities });
    },
    ...options,
  });
}

export function useUpdateActivity(
  options?: UseMutationOptions<Activity, Error, { id: string; data: Partial<Activity> }>
) {
  const queryClient = useQueryClient();

  return useMutation<Activity, Error, { id: string; data: Partial<Activity> }>({
    mutationFn: ({ id, data }) => apiClient.put<Activity>(`/activities/${id}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.activities });
    },
    ...options,
  });
}

export function useLeaderboard(
  timeframe: 'weekly' | 'monthly' = 'weekly',
  options?: ManagedQueryOptions<LeaderboardEntry[]>
) {
  return useQuery<LeaderboardEntry[]>({
    queryKey: queryKeys.leaderboard(timeframe),
    queryFn: () => apiClient.get<LeaderboardEntry[]>(`/leaderboard/${timeframe}`),
    ...options,
  });
}

export function useUserProfile(userId: string, options?: ManagedQueryOptions<any>) {
  return useQuery<any>({
    queryKey: queryKeys.userProfile(userId),
    queryFn: () => apiClient.get<any>(`/users/${userId}`),
    ...options,
  });
}

export function usePetProfile(petId: string, options?: ManagedQueryOptions<any>) {
  return useQuery<any>({
    queryKey: queryKeys.petProfile(petId),
    queryFn: () => apiClient.get<any>(`/pets/${petId}`),
    ...options,
  });
}

export function useCreatePet(options?: UseMutationOptions<any, Error, Record<string, unknown>>) {
  const queryClient = useQueryClient();

  return useMutation<any, Error, Record<string, unknown>>({
    mutationFn: async (data) => {
      const userId = useSessionStore.getState().user?.id;
      if (!userId) throw new Error('User not authenticated');

      return apiClient.post<any>('/pets', {
        ...data,
        owner: { connect: { id: userId } },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.friends });
      void useSessionStore.getState().refreshSession();
    },
    ...options,
  });
}

export function useFriends(options?: ManagedQueryOptions<any[]>) {
  return useQuery<any[]>({
    queryKey: queryKeys.friends,
    queryFn: () => apiClient.get<any[]>('/friends'),
    ...options,
  });
}

export function useAddFriend(options?: UseMutationOptions<void, Error, string>) {
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: (userId) => apiClient.post<void>(`/friends/${userId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.friends });
    },
    ...options,
  });
}

export function useConversations(options?: ManagedQueryOptions<any[]>) {
  return useQuery<any[]>({
    queryKey: queryKeys.messages,
    queryFn: () => apiClient.get<any[]>('/messages/conversations'),
    ...options,
  });
}

export function useConversation(conversationId: string, options?: ManagedQueryOptions<any>) {
  return useQuery<any>({
    queryKey: queryKeys.conversation(conversationId),
    queryFn: () => apiClient.get<any>(`/messages/conversations/${conversationId}`),
    enabled: !!conversationId,
    ...options,
  });
}

export function useSendMessage(
  options?: UseMutationOptions<any, Error, { conversationId: string; content: string }>
) {
  const queryClient = useQueryClient();

  return useMutation<any, Error, { conversationId: string; content: string }>({
    mutationFn: ({ conversationId, content }) =>
      apiClient.post<any>(`/messages/conversations/${conversationId}/messages`, { content }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.conversation(variables.conversationId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.messages });
    },
    ...options,
  });
}

export function useUploadImage(options?: UseMutationOptions<{ url: string }, Error, File>) {
  return useMutation<{ url: string }, Error, File>({
    mutationFn: async (file) => {
      const formData = new FormData();
      formData.append('file', file);
      return apiClient.upload<{ url: string }>('/upload/image', formData);
    },
    ...options,
  });
}

export function useSubmitQuiz(options?: UseMutationOptions<any, Error, any>) {
  const queryClient = useQueryClient();

  return useMutation<any, Error, any>({
    mutationFn: async (quizSession) => {
      const userId = useSessionStore.getState().user?.id;
      if (!userId) throw new Error('User not authenticated');

      return apiClient.post<any>('/quiz/submit', {
        ...quizSession,
        userId,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.userProfile(useSessionStore.getState().user?.id || ''),
      });
    },
    ...options,
  });
}

export function useGetMatches(options?: ManagedQueryOptions<SuggestedMatch[]>) {
  return useQuery<SuggestedMatch[]>({
    queryKey: ['matches'],
    queryFn: () => apiClient.get<SuggestedMatch[]>('/matches/suggested'),
    ...options,
  });
}

export function useRecordInteraction(
  options?: UseMutationOptions<
    any,
    Error,
    { targetUserId: string; action: 'like' | 'skip' | 'super_like' }
  >
) {
  return useMutation<any, Error, { targetUserId: string; action: 'like' | 'skip' | 'super_like' }>({
    mutationFn: (data) => apiClient.post<any>('/matches/interact', data),
    ...options,
  });
}
