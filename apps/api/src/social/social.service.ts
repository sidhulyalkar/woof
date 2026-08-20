import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { GamificationService } from '../gamification/gamification.service';
import { PrismaService } from '../prisma/prisma.service';
import { CreatePostDto, UpdatePostDto } from './dto/social.dto';

@Injectable()
export class SocialService {
  constructor(
    private prisma: PrismaService,
    private gamificationService: GamificationService,
  ) {}

  async createPost(userId: string, data: CreatePostDto) {
    await this.validateOwnedRelations(userId, data.petId, data.activityId);

    if (!data.text?.trim() && (!data.mediaUrls || data.mediaUrls.length === 0)) {
      throw new BadRequestException('A post needs text or media');
    }

    const post = await this.prisma.post.create({
      data: {
        author: { connect: { id: userId } },
        text: data.text?.trim() || null,
        mediaUrls: data.mediaUrls ?? [],
        visibility: data.visibility ?? 'PUBLIC',
        ...(data.petId ? { pet: { connect: { id: data.petId } } } : {}),
        ...(data.activityId ? { activity: { connect: { id: data.activityId } } } : {}),
      },
      include: this.postCardInclude(userId),
    });

    await this.gamificationService.awardPoints({
      userId,
      points: 2,
      reason: 'post_created',
      relatedEntityId: post.id,
    });

    return post;
  }

  async findAllPosts(
    viewerUserId: string,
    skip = 0,
    take = 20,
    authorUserId?: string,
    petId?: string,
  ) {
    const safeSkip = Math.max(0, Number(skip) || 0);
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 100));
    const where: Prisma.PostWhereInput = {};
    if (authorUserId) where.authorUserId = authorUserId;
    if (petId) where.petId = petId;

    const [posts, total] = await Promise.all([
      this.prisma.post.findMany({
        where,
        skip: safeSkip,
        take: safeTake,
        include: {
          ...this.postCardInclude(viewerUserId),
          activity: {
            select: {
              id: true,
              type: true,
              startedAt: true,
            },
          },
        },
        orderBy: { createdAt: 'desc' },
      }),
      this.prisma.post.count({ where }),
    ]);

    return { posts, total, skip: safeSkip, take: safeTake };
  }

  async findPostById(id: string, viewerUserId?: string) {
    const post = await this.prisma.post.findUnique({
      where: { id },
      include: {
        author: {
          select: { id: true, handle: true, avatarUrl: true, isVerified: true },
        },
        pet: {
          select: { id: true, name: true, species: true, breed: true, avatarUrl: true },
        },
        activity: {
          select: { id: true, type: true, startedAt: true, endedAt: true },
        },
        likes: viewerUserId
          ? { where: { userId: viewerUserId }, select: { id: true } }
          : {
              include: {
                user: { select: { id: true, handle: true, avatarUrl: true } },
              },
            },
        comments: {
          include: {
            user: {
              select: { id: true, handle: true, avatarUrl: true },
            },
          },
          orderBy: { createdAt: 'asc' },
        },
        _count: { select: { likes: true, comments: true } },
      },
    });

    if (!post) {
      throw new NotFoundException(`Post with ID ${id} not found`);
    }

    return post;
  }

  async updatePost(id: string, userId: string, data: UpdatePostDto) {
    await this.assertPostOwned(id, userId);
    await this.validateOwnedRelations(userId, data.petId, data.activityId);

    return this.prisma.post.update({
      where: { id },
      data: {
        ...(data.text !== undefined ? { text: data.text.trim() || null } : {}),
        ...(data.mediaUrls !== undefined ? { mediaUrls: data.mediaUrls } : {}),
        ...(data.visibility !== undefined ? { visibility: data.visibility } : {}),
        ...(data.petId !== undefined
          ? { pet: data.petId ? { connect: { id: data.petId } } : { disconnect: true } }
          : {}),
        ...(data.activityId !== undefined
          ? { activity: data.activityId ? { connect: { id: data.activityId } } : { disconnect: true } }
          : {}),
      },
      include: this.postCardInclude(userId),
    });
  }

  async deletePost(id: string, userId: string) {
    await this.assertPostOwned(id, userId);
    return this.prisma.post.delete({ where: { id } });
  }

  async createLike(postId: string, userId: string) {
    const post = await this.prisma.post.findUnique({
      where: { id: postId },
      select: { id: true },
    });
    if (!post) {
      throw new NotFoundException(`Post with ID ${postId} not found`);
    }

    const existingLike = await this.prisma.like.findUnique({
      where: { postId_userId: { postId, userId } },
    });

    if (existingLike) return existingLike;

    return this.prisma.like.create({
      data: {
        post: { connect: { id: postId } },
        user: { connect: { id: userId } },
      },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
    });
  }

  async deleteLike(postId: string, userId: string) {
    const existing = await this.prisma.like.findUnique({
      where: { postId_userId: { postId, userId } },
      select: { id: true },
    });
    if (!existing) return { success: true };

    await this.prisma.like.delete({
      where: { postId_userId: { postId, userId } },
    });
    return { success: true };
  }

  async getPostLikes(postId: string) {
    return this.prisma.like.findMany({
      where: { postId },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  async createComment(postId: string, userId: string, text: string) {
    const post = await this.prisma.post.findUnique({ where: { id: postId }, select: { id: true } });
    if (!post) throw new NotFoundException(`Post with ID ${postId} not found`);

    return this.prisma.comment.create({
      data: {
        text: text.trim(),
        post: { connect: { id: postId } },
        user: { connect: { id: userId } },
      },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
    });
  }

  async updateComment(id: string, userId: string, text: string) {
    await this.assertCommentOwned(id, userId);
    return this.prisma.comment.update({
      where: { id },
      data: { text: text.trim() },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
    });
  }

  async deleteComment(id: string, userId: string) {
    await this.assertCommentOwned(id, userId);
    return this.prisma.comment.delete({ where: { id } });
  }

  async getPostComments(postId: string) {
    return this.prisma.comment.findMany({
      where: { postId },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
      orderBy: { createdAt: 'asc' },
    });
  }

  private postCardInclude(viewerUserId?: string) {
    return {
      author: {
        select: { id: true, handle: true, avatarUrl: true, isVerified: true },
      },
      pet: {
        select: { id: true, name: true, species: true, breed: true, avatarUrl: true },
      },
      likes: viewerUserId
        ? { where: { userId: viewerUserId }, select: { id: true } }
        : { take: 0, select: { id: true } },
      _count: {
        select: { likes: true, comments: true },
      },
    } satisfies Prisma.PostInclude;
  }

  private async assertPostOwned(id: string, userId: string) {
    const post = await this.prisma.post.findFirst({
      where: { id, authorUserId: userId },
      select: { id: true },
    });
    if (!post) throw new NotFoundException(`Post with ID ${id} not found`);
  }

  private async assertCommentOwned(id: string, userId: string) {
    const comment = await this.prisma.comment.findFirst({
      where: { id, userId },
      select: { id: true },
    });
    if (!comment) throw new NotFoundException(`Comment with ID ${id} not found`);
  }

  private async validateOwnedRelations(userId: string, petId?: string, activityId?: string) {
    if (petId) {
      const pet = await this.prisma.pet.findFirst({
        where: { id: petId, ownerId: userId },
        select: { id: true },
      });
      if (!pet) throw new NotFoundException('Pet not found');
    }

    if (activityId) {
      const activity = await this.prisma.activity.findFirst({
        where: { id: activityId, userId },
        select: { id: true },
      });
      if (!activity) throw new NotFoundException('Activity not found');
    }
  }
}
