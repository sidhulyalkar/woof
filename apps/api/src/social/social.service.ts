import { Injectable, NotFoundException, ConflictException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { Prisma } from '@woof/database';

@Injectable()
export class SocialService {
  constructor(private prisma: PrismaService) {}

  // ============================================
  // POSTS
  // ============================================

  async createPost(data: Prisma.PostCreateInput) {
    return this.prisma.post.create({
      data,
      include: {
        author: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
        _count: {
          select: {
            likes: true,
            comments: true,
          },
        },
      },
    });
  }

  async findAllPosts(skip = 0, take = 20, authorUserId?: string, petId?: string) {
    const where: Prisma.PostWhereInput = {};
    if (authorUserId) where.authorUserId = authorUserId;
    if (petId) where.petId = petId;

    const [posts, total] = await Promise.all([
      this.prisma.post.findMany({
        where,
        skip,
        take,
        include: {
          author: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          pet: {
            select: {
              id: true,
              name: true,
              avatarUrl: true,
            },
          },
          activity: {
            select: {
              id: true,
              type: true,
              startedAt: true,
            },
          },
          _count: {
            select: {
              likes: true,
              comments: true,
            },
          },
        },
        orderBy: {
          createdAt: 'desc',
        },
      }),
      this.prisma.post.count({ where }),
    ]);

    return { posts, total, skip, take };
  }

  async findPostById(id: string) {
    const post = await this.prisma.post.findUnique({
      where: { id },
      include: {
        author: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
        pet: {
          select: {
            id: true,
            name: true,
            avatarUrl: true,
          },
        },
        activity: true,
        likes: {
          include: {
            user: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
          },
        },
        comments: {
          include: {
            user: {
              select: {
                id: true,
                handle: true,
                avatarUrl: true,
              },
            },
          },
          orderBy: {
            createdAt: 'asc',
          },
        },
      },
    });

    if (!post) {
      throw new NotFoundException(`Post with ID ${id} not found`);
    }

    return post;
  }

  async updatePost(id: string, data: Prisma.PostUpdateInput) {
    try {
      return await this.prisma.post.update({
        where: { id },
        data,
        include: {
          author: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
          _count: {
            select: {
              likes: true,
              comments: true,
            },
          },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Post with ID ${id} not found`);
      }
      throw error;
    }
  }

  async deletePost(id: string) {
    try {
      return await this.prisma.post.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Post with ID ${id} not found`);
      }
      throw error;
    }
  }

  // ============================================
  // LIKES
  // ============================================

  async createLike(postId: string, userId: string) {
    // Check if post exists
    const post = await this.prisma.post.findUnique({ where: { id: postId } });
    if (!post) {
      throw new NotFoundException(`Post with ID ${postId} not found`);
    }

    // Check if already liked
    const existingLike = await this.prisma.like.findUnique({
      where: {
        postId_userId: { postId, userId },
      },
    });

    if (existingLike) {
      throw new ConflictException('Post already liked by this user');
    }

    return this.prisma.like.create({
      data: {
        post: { connect: { id: postId } },
        user: { connect: { id: userId } },
      },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async deleteLike(postId: string, userId: string) {
    try {
      return await this.prisma.like.delete({
        where: {
          postId_userId: { postId, userId },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException('Like not found');
      }
      throw error;
    }
  }

  async getPostLikes(postId: string) {
    return this.prisma.like.findMany({
      where: { postId },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: {
        createdAt: 'desc',
      },
    });
  }

  // ============================================
  // COMMENTS
  // ============================================

  async createComment(data: Prisma.CommentCreateInput) {
    return this.prisma.comment.create({
      data,
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
    });
  }

  async updateComment(id: string, text: string) {
    try {
      return await this.prisma.comment.update({
        where: { id },
        data: { text },
        include: {
          user: {
            select: {
              id: true,
              handle: true,
              avatarUrl: true,
            },
          },
        },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Comment with ID ${id} not found`);
      }
      throw error;
    }
  }

  async deleteComment(id: string) {
    try {
      return await this.prisma.comment.delete({
        where: { id },
      });
    } catch (error) {
      if (error.code === 'P2025') {
        throw new NotFoundException(`Comment with ID ${id} not found`);
      }
      throw error;
    }
  }

  async getPostComments(postId: string) {
    return this.prisma.comment.findMany({
      where: { postId },
      include: {
        user: {
          select: {
            id: true,
            handle: true,
            avatarUrl: true,
          },
        },
      },
      orderBy: {
        createdAt: 'asc',
      },
    });
  }
}
