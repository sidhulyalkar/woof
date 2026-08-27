import {
  Body,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Put,
  Query,
  Request,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiResponse, ApiTags } from '@nestjs/swagger';
import type { AuthenticatedRequest } from '../auth/authenticated-request';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';
import { CreateCommentDto, CreatePostDto, UpdateCommentDto, UpdatePostDto } from './dto/social.dto';
import { SocialService } from './social.service';

@ApiTags('social')
@Controller('social')
@UseGuards(JwtAuthGuard)
@ApiBearerAuth()
export class SocialController {
  constructor(private socialService: SocialService) {}

  @Post('posts')
  @ApiOperation({ summary: 'Create a post as the authenticated user' })
  @ApiResponse({ status: 201, description: 'Post created successfully' })
  async createPost(@Request() req: AuthenticatedRequest, @Body() dto: CreatePostDto) {
    return this.socialService.createPost(req.user.sub, dto);
  }

  @Get('posts')
  @ApiOperation({ summary: 'Get privacy-authorized posts (paginated)' })
  async findAllPosts(
    @Request() req: AuthenticatedRequest,
    @Query('skip') skip?: number,
    @Query('take') take?: number,
    @Query('authorUserId') authorUserId?: string,
    @Query('petId') petId?: string
  ) {
    return this.socialService.findAllPosts(req.user.sub, skip, take, authorUserId, petId);
  }

  @Get('posts/:id')
  @ApiOperation({ summary: 'Get a privacy-authorized post by ID' })
  async findOnePost(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.socialService.findPostById(id, req.user.sub);
  }

  @Put('posts/:id')
  @ApiOperation({ summary: 'Update a post owned by the authenticated user' })
  async updatePost(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body() dto: UpdatePostDto
  ) {
    return this.socialService.updatePost(id, req.user.sub, dto);
  }

  @Delete('posts/:id')
  @ApiOperation({ summary: 'Delete a post owned by the authenticated user' })
  async deletePost(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.socialService.deletePost(id, req.user.sub);
  }

  @Post('posts/:postId/likes')
  @ApiOperation({ summary: 'Like a visible post as the authenticated user' })
  async createLike(@Request() req: AuthenticatedRequest, @Param('postId') postId: string) {
    return this.socialService.createLike(postId, req.user.sub);
  }

  @Delete('posts/:postId/likes')
  @ApiOperation({ summary: 'Remove the authenticated user’s like from a post' })
  async deleteLike(@Request() req: AuthenticatedRequest, @Param('postId') postId: string) {
    return this.socialService.deleteLike(postId, req.user.sub);
  }

  @Get('posts/:postId/likes')
  @ApiOperation({ summary: 'Get visible likes for a privacy-authorized post' })
  async getPostLikes(@Request() req: AuthenticatedRequest, @Param('postId') postId: string) {
    return this.socialService.getPostLikes(postId, req.user.sub);
  }

  @Post('posts/:postId/comments')
  @ApiOperation({ summary: 'Comment on a visible post as the authenticated user' })
  async createComment(
    @Request() req: AuthenticatedRequest,
    @Param('postId') postId: string,
    @Body() dto: CreateCommentDto
  ) {
    return this.socialService.createComment(postId, req.user.sub, dto.text);
  }

  @Get('posts/:postId/comments')
  @ApiOperation({ summary: 'Get visible comments for a privacy-authorized post' })
  async getPostComments(@Request() req: AuthenticatedRequest, @Param('postId') postId: string) {
    return this.socialService.getPostComments(postId, req.user.sub);
  }

  @Put('comments/:id')
  @ApiOperation({ summary: 'Update a comment owned by the authenticated user' })
  async updateComment(
    @Request() req: AuthenticatedRequest,
    @Param('id') id: string,
    @Body() dto: UpdateCommentDto
  ) {
    return this.socialService.updateComment(id, req.user.sub, dto.text);
  }

  @Delete('comments/:id')
  @ApiOperation({ summary: 'Delete a comment owned by the authenticated user' })
  async deleteComment(@Request() req: AuthenticatedRequest, @Param('id') id: string) {
    return this.socialService.deleteComment(id, req.user.sub);
  }
}
