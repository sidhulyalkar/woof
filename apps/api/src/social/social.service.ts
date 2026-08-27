import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { HouseholdsService } from '../households/households.service';
import { PrismaService } from '../prisma/prisma.service';
import { CreatePostDto, UpdatePostDto } from './dto/social.dto';

@Injectable()
export class SocialService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly households: HouseholdsService,
  ) {}

  async createPost(userId: string, data: CreatePostDto) {
    await this.validateAccessibleRelations(userId, data.petId, data.activityId);

    if (!data.text?.trim() && (!data.mediaUrls || data.mediaUrls.length === 0)) {
      throw new BadRequestException('A post needs text or media');
    }

    return this.prisma.post.create({
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
    const blockedIds = await this.blockedUserIds(viewerUserId);
    const filters: Prisma.PostWhereInput = {};
    if (authorUserId) filters.authorUserId = authorUserId;
    if (petId) filters.petId = petId;

    const where: Prisma.PostWhereInput = {
      AND: [filters, this.visiblePostWhere(viewerUserId, blockedIds)],
    };

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

  async findPostById(id: string, viewerUserId: string) {
    const blockedIds = await this.blockedUserIds(viewerUserId);
    const post = await this.prisma.post.findFirst({
      where: {
        id,
        AND: [this.visiblePostWhere(viewerUserId, blockedIds)],
      },
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
        likes: { where: { userId: viewerUserId }, select: { id: true } },
        comments: {
          where: blockedIds.length ? { userId: { notIn: blockedIds } } : undefined,
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
    await this.validateAccessibleRelations(userId, data.petId, data.activityId);

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
    await this.assertPostViewable(postId, userId);

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

  async getPostLikes(postId: string, viewerUserId: string) {
    await this.assertPostViewable(postId, viewerUserId);
    const blockedIds = await this.blockedUserIds(viewerUserId);
    return this.prisma.like.findMany({
      where: {
        postId,
        ...(blockedIds.length ? { userId: { notIn: blockedIds } } : {}),
      },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
      orderBy: { createdAt: 'desc' },
    });
  }

  async createComment(postId: string, userId: string, text: string) {
    await this.assertPostViewable(postId, userId);

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

  async getPostComments(postId: string, viewerUserId: string) {
    await this.assertPostViewable(postId, viewerUserId);
    const blockedIds = await this.blockedUserIds(viewerUserId);
    return this.prisma.comment.findMany({
      where: {
        postId,
        ...(blockedIds.length ? { userId: { notIn: blockedIds } } : {}),
      },
      include: {
        user: { select: { id: true, handle: true, avatarUrl: true } },
      },
      orderBy: { createdAt: 'asc' },
    });
  }

  private postCardInclude(viewerUserId: string) {
    return {
      author: {
        select: { id: true, handle: true, avatarUrl: true, isVerified: true },
      },
      pet: {
        select: { id: true, name: true, species: true, breed: true, avatarUrl: true },
      },
      likes: { where: { userId: viewerUserId }, select: { id: true } },
      _count: {
        select: { likes: true, comments: true },
      },
    } satisfies Prisma.PostInclude;
  }

  private visiblePostWhere(viewerUserId: string, blockedIds: string[]): Prisma.PostWhereInput {
    return {
      AND: [
        {
          OR: [{ authorUserId: viewerUserId }, { visibility: 'PUBLIC' }],
        },
        ...(blockedIds.length ? [{ authorUserId: { notIn: blockedIds } }] : []),
      ],
    };
  }

  private async assertPostOwned(id: string, userId: string) {
    const post = await this.prisma.post.findFirst({
      where: { id, authorUserId: userId },
      select: { id: true },
    });
    if (!post) throw new NotFoundException(`Post with ID ${id} not found`);
  }

  private async assertPostViewable(id: string, userId: string) {
    const blockedIds = await this.blockedUserIds(userId);
    const post = await this.prisma.post.findFirst({
      where: {
        id,
        AND: [this.visiblePostWhere(userId, blockedIds)],
      },
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

  private async validateAccessibleRelations(userId: string, petId?: string, activityId?: string) {
    if (petId) {
      await this.households.assertPetAccessible(userId, petId);
    }

    if (activityId) {
      const activity = await this.prisma.activity.findFirst({
        where: {
          id: activityId,
          AND: [this.households.householdActivityWhere(userId)],
        },
        select: { id: true },
      });
      if (!activity) throw new NotFoundException('Activity not found');
    }
  }

  private async blockedUserIds(userId: string) {
    const rows = await this.prisma.blockedUser.findMany({
      where: {
        OR: [{ userId }, { blockedId: userId }],
      },
      select: { userId: true, blockedId: true },
    });

    return [...new Set(rows.map((row) => (row.userId === userId ? row.blockedId : row.userId)))];
  }
}
