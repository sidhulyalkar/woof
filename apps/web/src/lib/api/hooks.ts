import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
  type UseQueryOptions,
} from '@tanstack/react-query';
import type { MLFeatureVector, QuizSession } from '@/types/quiz';
import { useSessionStore } from '@/store/session';
import { apiClient } from './client';

type ManagedQueryOptions<T> = Omit<
  UseQueryOptions<T, Error, T, readonly unknown[]>,
  'queryKey' | 'queryFn'
>;

type ExtensibleRecord = Record<string, unknown>;

type BackendIdentity = {
  id?: string;
  handle?: string;
  avatarUrl?: string | null;
};

type BackendPost = {
  id: string;
  authorUserId: string;
  petId?: string | null;
  activityId?: string | null;
  text?: string | null;
  mediaUrls?: string[] | null;
  createdAt: string;
  _count?: {
    likes?: number;
    comments?: number;
  };
  author?: BackendIdentity | null;
  pet?: {
    id: string;
    name: string;
    avatarUrl?: string | null;
  } | null;
};

type BackendPostList = BackendPost[] | { posts?: BackendPost[] };

type BackendComment = {
  id: string;
  postId: string;
  authorUserId?: string;
  userId?: string;
  text: string;
  createdAt: string;
  author?: BackendIdentity | null;
  user?: BackendIdentity | null;
};

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

type ActivityListResponse =
  | Activity[]
  | {
      activities: Activity[];
      total?: number;
      skip?: number;
      take?: number;
    };

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

export interface UserProfile extends ExtensibleRecord {
  id: string;
  handle: string;
  email?: string;
  bio?: string | null;
  avatarUrl?: string | null;
}

export interface PetProfile extends ExtensibleRecord {
  id: string;
  name: string;
  species: string;
  breed?: string | null;
  avatarUrl?: string | null;
}

export interface Friend extends ExtensibleRecord {
  id: string;
  handle?: string;
  avatarUrl?: string | null;
}

export interface ConversationSummary extends ExtensibleRecord {
  id: string;
}

export interface ConversationMessage extends ExtensibleRecord {
  id: string;
  content?: string;
}

export interface ConversationDetail extends ExtensibleRecord {
  id: string;
  messages?: ConversationMessage[];
}

type SubmitQuizInput = {
  session: QuizSession;
  featureVector: MLFeatureVector;
};

type QuizSaveResponse = {
  id: string;
  petId: string | null;
  sessionId: string;
  completedAt: string;
};

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

function transformBackendPost(backendPost: BackendPost): Post {
  return {
    id: backendPost.id,
    userId: backendPost.authorUserId,
    petId: backendPost.petId ?? undefined,
    activityId: backendPost.activityId ?? undefined,
    content: backendPost.text ?? '',
    images: backendPost.mediaUrls ?? [],
    likes: backendPost._count?.likes ?? 0,
    comments: backendPost._count?.comments ?? 0,
    createdAt: backendPost.createdAt,
    user: {
      id: backendPost.author?.id ?? backendPost.authorUserId,
      username: backendPost.author?.handle ?? 'Unknown',
      avatar: backendPost.author?.avatarUrl ?? undefined,
    },
    pet: backendPost.pet
      ? {
          id: backendPost.pet.id,
          name: backendPost.pet.name,
          avatar: backendPost.pet.avatarUrl ?? undefined,
        }
      : undefined,
  };
}

function transformBackendComment(comment: BackendComment): Comment {
  const author = comment.author ?? comment.user;
  const authorUserId = comment.authorUserId ?? comment.userId ?? author?.id;
  if (!authorUserId) {
    throw new Error('Comment response is missing its authenticated author identity');
  }

  return {
    id: comment.id,
    postId: comment.postId,
    authorUserId,
    text: comment.text,
    createdAt: comment.createdAt,
    author: author
      ? {
          id: author.id ?? authorUserId,
          handle: author.handle ?? 'Unknown',
          avatarUrl: author.avatarUrl ?? undefined,
        }
      : undefined,
  };
}

export function useFeed(options?: ManagedQueryOptions<Post[]>) {
  return useQuery<Post[]>({
    queryKey: queryKeys.feed,
    queryFn: async () => {
      const response = await apiClient.get<BackendPostList>('/social/posts?skip=0&take=20');
      const posts = Array.isArray(response) ? response : (response.posts ?? []);
      return posts.map(transformBackendPost);
    },
    ...options,
  });
}

export function usePost(postId: string, options?: ManagedQueryOptions<Post>) {
  return useQuery<Post>({
    queryKey: ['post', postId],
    queryFn: async () => {
      const response = await apiClient.get<BackendPost>(`/social/posts/${postId}`);
      return transformBackendPost(response);
    },
    ...options,
  });
}

export function useCreatePost(options?: UseMutationOptions<Post, Error, Partial<Post>>) {
  const queryClient = useQueryClient();

  return useMutation<Post, Error, Partial<Post>>({
    mutationFn: async (data) => {
      const payload = {
        text: data.content ?? '',
        mediaUrls: data.images ?? [],
        petId: data.petId,
        activityId: data.activityId,
      };

      const result = await apiClient.post<BackendPost>('/social/posts', payload);
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
    mutationFn: (postId) => apiClient.post<void>(`/social/posts/${postId}/likes`),
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
      const response = await apiClient.get<BackendComment[]>(`/social/posts/${postId}/comments`);
      return response.map(transformBackendComment);
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
      const response = await apiClient.post<BackendComment>(`/social/posts/${postId}/comments`, {
        text,
      });
      return transformBackendComment(response);
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
    queryFn: async () => {
      const response = await apiClient.get<ActivityListResponse>('/activities');
      return Array.isArray(response) ? response : response.activities;
    },
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

export function useUserProfile(userId: string, options?: ManagedQueryOptions<UserProfile>) {
  return useQuery<UserProfile>({
    queryKey: queryKeys.userProfile(userId),
    queryFn: () => apiClient.get<UserProfile>(`/users/${userId}`),
    ...options,
  });
}

export function usePetProfile(petId: string, options?: ManagedQueryOptions<PetProfile>) {
  return useQuery<PetProfile>({
    queryKey: queryKeys.petProfile(petId),
    queryFn: () => apiClient.get<PetProfile>(`/pets/${petId}`),
    ...options,
  });
}

export function useCreatePet(
  options?: UseMutationOptions<PetProfile, Error, Record<string, unknown>>
) {
  const queryClient = useQueryClient();

  return useMutation<PetProfile, Error, Record<string, unknown>>({
    mutationFn: (data) => apiClient.post<PetProfile>('/pets', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.friends });
      void useSessionStore.getState().refreshSession();
    },
    ...options,
  });
}

export function useFriends(options?: ManagedQueryOptions<Friend[]>) {
  return useQuery<Friend[]>({
    queryKey: queryKeys.friends,
    queryFn: () => apiClient.get<Friend[]>('/friends'),
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

export function useConversations(options?: ManagedQueryOptions<ConversationSummary[]>) {
  return useQuery<ConversationSummary[]>({
    queryKey: queryKeys.messages,
    queryFn: () => apiClient.get<ConversationSummary[]>('/messages/conversations'),
    ...options,
  });
}

export function useConversation(
  conversationId: string,
  options?: ManagedQueryOptions<ConversationDetail>
) {
  return useQuery<ConversationDetail>({
    queryKey: queryKeys.conversation(conversationId),
    queryFn: () => apiClient.get<ConversationDetail>(`/messages/conversations/${conversationId}`),
    enabled: Boolean(conversationId),
    ...options,
  });
}

export function useSendMessage(
  options?: UseMutationOptions<
    ConversationMessage,
    Error,
    { conversationId: string; content: string }
  >
) {
  const queryClient = useQueryClient();

  return useMutation<ConversationMessage, Error, { conversationId: string; content: string }>({
    mutationFn: ({ conversationId, content }) =>
      apiClient.post<ConversationMessage>(`/messages/conversations/${conversationId}/messages`, {
        content,
      }),
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

export function useSubmitQuiz(
  options?: UseMutationOptions<QuizSaveResponse, Error, SubmitQuizInput>
) {
  const queryClient = useQueryClient();

  return useMutation<QuizSaveResponse, Error, SubmitQuizInput>({
    mutationFn: async ({ session }) => {
      const responses = Object.fromEntries(
        session.responses.map(({ questionId, answer }) => [questionId, answer])
      );

      return apiClient.post<QuizSaveResponse>('/quiz/responses', {
        sessionId: session.id,
        petId: session.petId,
        responses,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: queryKeys.userProfile(useSessionStore.getState().user?.id ?? ''),
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
    ExtensibleRecord,
    Error,
    { targetUserId: string; action: 'like' | 'skip' | 'super_like' }
  >
) {
  return useMutation<
    ExtensibleRecord,
    Error,
    { targetUserId: string; action: 'like' | 'skip' | 'super_like' }
  >({
    mutationFn: (data) => apiClient.post<ExtensibleRecord>('/matches/interact', data),
    ...options,
  });
}
