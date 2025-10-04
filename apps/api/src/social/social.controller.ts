import { Controller, Get, Post, Put, Delete, Param, Body, Query, UseGuards } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { SocialService } from './social.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { Prisma } from '@woof/database';

@ApiTags('social')
@Controller('social')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class SocialController {
  constructor(private socialService: SocialService) {}

  // ============================================
  // POSTS
  // ============================================

  @Post('posts')
  @ApiOperation({ summary: 'Create a new post' })
  @ApiResponse({ status: 201, description: 'Post created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async createPost(@Body() createPostDto: Prisma.PostCreateInput) {
    return this.socialService.createPost(createPostDto);
  }

  @Get('posts')
  @ApiOperation({ summary: 'Get all posts (paginated)' })
  @ApiResponse({ status: 200, description: 'List of posts' })
  async findAllPosts(
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('authorUserId') authorUserId?: string,
    @Query('petId') petId?: string,
  ) {
    // Ensure proper number conversion with defaults
    const skipNum = skip !== undefined ? Number(skip) : 0;
    const takeNum = take !== undefined ? Number(take) : 20;
    return this.socialService.findAllPosts(skipNum, takeNum, authorUserId, petId);
  }

  @Get('posts/:id')
  @ApiOperation({ summary: 'Get post by ID' })
  @ApiResponse({ status: 200, description: 'Post found' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  async findOnePost(@Param('id') id: string) {
    return this.socialService.findPostById(id);
  }

  @Put('posts/:id')
  @ApiOperation({ summary: 'Update post by ID' })
  @ApiResponse({ status: 200, description: 'Post updated successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  async updatePost(
    @Param('id') id: string,
    @Body() updatePostDto: Prisma.PostUpdateInput,
  ) {
    return this.socialService.updatePost(id, updatePostDto);
  }

  @Delete('posts/:id')
  @ApiOperation({ summary: 'Delete post by ID' })
  @ApiResponse({ status: 200, description: 'Post deleted successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  async deletePost(@Param('id') id: string) {
    return this.socialService.deletePost(id);
  }

  // ============================================
  // LIKES
  // ============================================

  @Post('posts/:postId/likes')
  @ApiOperation({ summary: 'Like a post' })
  @ApiResponse({ status: 201, description: 'Post liked successfully' })
  @ApiResponse({ status: 404, description: 'Post not found' })
  @ApiResponse({ status: 409, description: 'Post already liked' })
  async createLike(
    @Param('postId') postId: string,
    @Body('userId') userId: string,
  ) {
    return this.socialService.createLike(postId, userId);
  }

  @Delete('posts/:postId/likes/:userId')
  @ApiOperation({ summary: 'Unlike a post' })
  @ApiResponse({ status: 200, description: 'Post unliked successfully' })
  @ApiResponse({ status: 404, description: 'Like not found' })
  async deleteLike(
    @Param('postId') postId: string,
    @Param('userId') userId: string,
  ) {
    return this.socialService.deleteLike(postId, userId);
  }

  @Get('posts/:postId/likes')
  @ApiOperation({ summary: 'Get all likes for a post' })
  @ApiResponse({ status: 200, description: 'List of likes' })
  async getPostLikes(@Param('postId') postId: string) {
    return this.socialService.getPostLikes(postId);
  }

  // ============================================
  // COMMENTS
  // ============================================

  @Post('posts/:postId/comments')
  @ApiOperation({ summary: 'Create a comment on a post' })
  @ApiResponse({ status: 201, description: 'Comment created successfully' })
  @ApiResponse({ status: 400, description: 'Invalid input' })
  async createComment(
    @Param('postId') postId: string,
    @Body() createCommentDto: Prisma.CommentCreateInput,
  ) {
    // Ensure the post connection is properly set using the postId from the URL
    const commentData: Prisma.CommentCreateInput = {
      ...createCommentDto,
      post: { connect: { id: postId } },
    };
    return this.socialService.createComment(commentData);
  }

  @Get('posts/:postId/comments')
  @ApiOperation({ summary: 'Get all comments for a post' })
  @ApiResponse({ status: 200, description: 'List of comments' })
  async getPostComments(@Param('postId') postId: string) {
    return this.socialService.getPostComments(postId);
  }

  @Put('comments/:id')
  @ApiOperation({ summary: 'Update comment by ID' })
  @ApiResponse({ status: 200, description: 'Comment updated successfully' })
  @ApiResponse({ status: 404, description: 'Comment not found' })
  async updateComment(
    @Param('id') id: string,
    @Body('text') text: string,
  ) {
    return this.socialService.updateComment(id, text);
  }

  @Delete('comments/:id')
  @ApiOperation({ summary: 'Delete comment by ID' })
  @ApiResponse({ status: 200, description: 'Comment deleted successfully' })
  @ApiResponse({ status: 404, description: 'Comment not found' })
  async deleteComment(@Param('id') id: string) {
    return this.socialService.deleteComment(id);
  }
}
